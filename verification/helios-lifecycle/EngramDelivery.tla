----------------------- MODULE EngramDelivery -----------------------
EXTENDS Naturals, Sequences, FiniteSets, TLC

\* Bounded model of one disk-backed embedding table's delivery protocol: the
\* two staging frames fenced by tensor-parallel ranks, the bounded submission
\* channel, the graph's publication of a step's row ids, and the bounded advisory
\* prefetch channel that warms the host row cache without owning a frame. The
\* reader thread takes a submission as soon as it arrives but gathers it only
\* once the step's graph published its ids, and warms the cache while it waits.
\* It checks that no rank, frame, or channel can deadlock, that an advisory batch
\* can never disturb a frame's generation, that warming never starts while a
\* readable gather waits, and that the reader thread cannot retire while real
\* gather work is still queued or held.
\*
\* Correspondence: State::prepare and State::finish plus their rejection
\* messages, Shared::start's blocking SyncSender::send, Requests::ready,
\* CachedReader::gather and CachedReader::prefetch, next_work and the worker loop
\* in Shared::with_readers, and Transfers::enqueue's wait_value on the frame's
\* ready word. Like RankDispatch this is a standalone protocol model, not a
\* refinement proof of the threads. Publication is not ordered between steps
\* here, which only adds behaviours: on the device a step's graph starts after
\* the previous step's graph finishes.

CONSTANTS
    \* tensor-parallel ranks sharing one table
    RankCount,
    \* staging frames per table, STAGING_FRAMES
    FrameCount,
    \* depth of the submission channel
    RealCapacity,
    \* depth of the advisory prefetch channel
    AdvisoryCapacity,
    \* distinct submission sequences explored
    MaxSequence,
    \* distinct row counts explored
    MaxRows,
    \* advisory batches the environment may hand over
    MaxOffers

\* The short receive timeout that lets a parked reader thread still observe
\* queued advisory work. Shipped enabled, and the mutation list checks that a
\* thread which never wakes leaves advisory work unhandled.
PollWakeup == TRUE

VARIABLES state

Frames == 0..(FrameCount - 1)
Ranks == 0..(RankCount - 1)
Sequences == 1..MaxSequence
RowCounts == 1..MaxRows
NoJob == [frame |-> (0 - 1), gen |-> (0 - 1)]
Real == "real"
Advisory == "advisory"
Idle == "idle"

FrameOf(seq) == seq % FrameCount

Initial ==
    [ stateSeq        |-> [f \in Frames |-> 0],
      generation      |-> [f \in Frames |-> 0],
      submitted       |-> [f \in Frames |-> {}],
      completed       |-> [f \in Frames |-> {}],
      rowCount        |-> [f \in Frames |-> 0],
      published       |-> [f \in Frames |-> 0],
      waiting         |-> [f \in Frames |-> {}],
      idsReady        |-> [f \in Frames |-> 0],
      queue           |-> <<>>,
      advisoryPending |-> 0,
      offers          |-> 0,
      held            |-> NoJob,
      job             |-> NoJob,
      worker          |-> Idle,
      parked          |-> FALSE,
      entryBacklog    |-> FALSE,
      exited          |-> FALSE,
      realClosed      |-> FALSE,
      advisoryClosed  |-> FALSE ]

Init == state = Initial

\* Shared::start. A fresh sequence may occupy a frame slot only once every rank
\* retired the previous occupant, and it enqueues exactly one gather job. A
\* second rank submitting the same sequence adds its bit without a new job.
\* A submission into a full channel blocks the rank until the reader thread
\* pops an entry, so the action is simply not enabled while the channel is full.
Submit(before, after, rank, seq, rows) ==
    LET f == FrameOf(seq)
        fresh == before.stateSeq[f] # seq
    IN /\ ~before.exited
       /\ rank \in Ranks
       /\ seq \in Sequences
       /\ rows \in RowCounts
       /\ IF fresh
            THEN /\ seq > before.stateSeq[f]
                 /\ (before.stateSeq[f] = 0 \/ before.completed[f] = Ranks)
                 /\ Len(before.queue) < RealCapacity
            ELSE /\ rank \notin before.submitted[f]
                 /\ before.rowCount[f] = rows
       /\ after = [before EXCEPT
             !.submitted[f] = before.submitted[f] \cup {rank},
             !.stateSeq[f] = IF fresh THEN seq ELSE before.stateSeq[f],
             !.generation[f] = IF fresh THEN before.generation[f] + 1
                               ELSE before.generation[f],
             !.completed[f] = IF fresh THEN {} ELSE before.completed[f],
             !.published[f] = IF fresh THEN 0 ELSE before.published[f],
             !.waiting[f] = IF fresh THEN {} ELSE before.waiting[f],
             !.idsReady[f] = IF fresh THEN 0 ELSE before.idsReady[f],
             !.rowCount[f] = IF fresh THEN rows ELSE before.rowCount[f],
             !.queue = IF fresh
                         THEN Append(before.queue,
                                     [frame |-> f, gen |-> before.generation[f] + 1])
                         ELSE before.queue]

\* Transfers::enqueue registers the rank's copy stream wait on the frame's
\* ready word for the generation it was handed.
GraphWait(before, after, rank, seq) ==
    LET f == FrameOf(seq)
    IN /\ ~before.exited
       /\ rank \in Ranks
       /\ seq \in Sequences
       /\ before.stateSeq[f] = seq
       /\ rank \in before.submitted[f]
       /\ rank \notin before.waiting[f]
       /\ after = [before EXCEPT !.waiting[f] = before.waiting[f] \cup {rank}]

\* The CUDA stream observation releases only once the reader thread published
\* that frame's generation.
GraphProceed(before, after, rank, seq) ==
    LET f == FrameOf(seq)
    IN /\ ~before.exited
       /\ rank \in before.waiting[f]
       /\ before.stateSeq[f] = seq
       /\ before.published[f] = before.generation[f]
       /\ before.published[f] # 0
       /\ after = [before EXCEPT !.waiting[f] = before.waiting[f] \ {rank}]

\* Shared::finish. Retirement follows the transfer event, so the rank's wait
\* must already have been released and the rows must have been published.
Retire(before, after, rank, seq) ==
    LET f == FrameOf(seq)
    IN /\ ~before.exited
       /\ rank \in Ranks
       /\ seq \in Sequences
       /\ before.stateSeq[f] = seq
       /\ rank \in before.submitted[f]
       /\ rank \notin before.completed[f]
       /\ rank \notin before.waiting[f]
       /\ before.published[f] = before.generation[f]
       /\ before.published[f] # 0
       /\ after = [before EXCEPT !.completed[f] = before.completed[f] \cup {rank}]

\* The step's graph writes its row ids to the mailbox and publishes the sequence.
\* It runs once a rank launched it, independently of the reader thread.
PublishIds(before, after, f) ==
    /\ ~before.exited
    /\ f \in Frames
    /\ before.stateSeq[f] # 0
    /\ before.submitted[f] # {}
    /\ before.idsReady[f] # before.generation[f]
    /\ after = [before EXCEPT !.idsReady[f] = before.generation[f]]

\* next_work takes the oldest submission into its single held slot as soon as
\* one arrives, whether or not its ids exist yet.
Hold(before, after) ==
    /\ ~before.exited
    /\ before.worker = Idle
    /\ before.held = NoJob
    /\ Len(before.queue) # 0
    /\ after = [before EXCEPT
          !.held = Head(before.queue),
          !.queue = Tail(before.queue),
          !.parked = FALSE]

\* A gather the thread could start now: a submission it has not taken yet, or a
\* held one whose ids the graph already published.
GatherWaiting(s) ==
    \/ (s.held = NoJob /\ Len(s.queue) # 0)
    \/ (s.held # NoJob /\ s.idsReady[s.held.frame] = s.held.gen)

\* The reader thread always prefers a readable gather job.
BeginReal(before, after) ==
    /\ ~before.exited
    /\ before.worker = Idle
    /\ before.held # NoJob
    /\ before.idsReady[before.held.frame] = before.held.gen
    /\ after = [before EXCEPT
          !.worker = Real,
          !.job = before.held,
          !.held = NoJob]

CompleteReal(before, after) ==
    /\ before.worker = Real
    /\ after = [before EXCEPT
          !.worker = Idle,
          !.job = NoJob,
          !.published = [f \in Frames |->
                            IF f = before.job.frame
                                THEN before.job.gen
                                ELSE before.published[f]]]

\* Advisory work is taken only while no gather could start: either nothing is
\* submitted, or the held submission's graph has not published its ids. The
\* second case is the replay window read-ahead exists to use. `entryBacklog`
\* records whether a gather could start at the moment of entry, so the priority
\* decision is checkable after a submission or publication has followed it.
BeginAdvisory(before, after) ==
    /\ ~before.exited
    /\ before.worker = Idle
    /\ ~before.parked
    /\ ~GatherWaiting(before)
    /\ before.advisoryPending # 0
    /\ after = [before EXCEPT
          !.worker = Advisory,
          !.entryBacklog = GatherWaiting(before),
          !.advisoryPending = before.advisoryPending - 1]

CompleteAdvisory(before, after) ==
    /\ before.worker = Advisory
    /\ after = [before EXCEPT !.worker = Idle, !.entryBacklog = FALSE]

\* Shared::prefetch hands a chunk over the bounded channel without blocking.
OfferAdvisory(before, after) ==
    /\ ~before.exited
    /\ ~before.advisoryClosed
    /\ before.offers < MaxOffers
    /\ before.advisoryPending < AdvisoryCapacity
    /\ after = [before EXCEPT
          !.advisoryPending = before.advisoryPending + 1,
          !.offers = before.offers + 1]

\* An offer that arrives at capacity is discarded: never queued, never blocking.
DropAdvisory(before, after) ==
    /\ ~before.exited
    /\ ~before.advisoryClosed
    /\ before.offers < MaxOffers
    /\ before.advisoryPending = AdvisoryCapacity
    /\ after = [before EXCEPT !.offers = before.offers + 1]

\* Shared::with_readers. With nothing held or queued the thread parks on its wake
\* token. A parked thread still wakes for a real submission because Hold clears
\* the park flag; only the receive timeout wakes it for advisory work. A held
\* submission is never parked on: its ids are observed by polling.
Park(before, after) ==
    /\ ~before.exited
    /\ before.worker = Idle
    /\ ~before.parked
    /\ before.held = NoJob
    /\ Len(before.queue) = 0
    /\ after = [before EXCEPT !.parked = TRUE]

WakeByPoll(before, after) ==
    /\ before.parked
    /\ PollWakeup
    /\ Len(before.queue) = 0
    /\ before.advisoryPending # 0
    /\ after = [before EXCEPT !.parked = FALSE]

\* A parked thread that its receive timeout wakes can also take the batch in the
\* same step. This is the action that makes the timeout load-bearing: with
\* PollWakeup disabled, a parked thread can never reach advisory work.
WakeAndTake(before, after) ==
    /\ ~before.exited
    /\ before.worker = Idle
    /\ before.parked
    /\ PollWakeup
    /\ Len(before.queue) = 0
    /\ before.advisoryPending # 0
    /\ after = [before EXCEPT
          !.worker = Advisory,
          !.parked = FALSE,
          !.entryBacklog = FALSE,
          !.advisoryPending = before.advisoryPending - 1]

\* Layer::Drop releases both senders and then joins the thread.
CloseReal(before, after) ==
    /\ ~before.realClosed
    /\ after = [before EXCEPT !.realClosed = TRUE]

CloseAdvisory(before, after) ==
    /\ ~before.advisoryClosed
    /\ after = [before EXCEPT !.advisoryClosed = TRUE]

\* The thread exits only when the submission channel is closed and drained. An
\* advisory-channel disconnect is not a shutdown signal: exiting on it would
\* strand a queued gather whose frame every rank is waiting on.
Exit(before, after) ==
    /\ ~before.exited
    /\ before.worker = Idle
    /\ before.realClosed
    /\ before.held = NoJob
    /\ Len(before.queue) = 0
    /\ after = [before EXCEPT !.exited = TRUE, !.parked = FALSE]

Next ==
    \/ (\E rank \in Ranks, seq \in Sequences, rows \in RowCounts :
           Submit(state, state', rank, seq, rows))
    \/ (\E rank \in Ranks, seq \in Sequences : GraphWait(state, state', rank, seq))
    \/ (\E rank \in Ranks, seq \in Sequences : GraphProceed(state, state', rank, seq))
    \/ (\E rank \in Ranks, seq \in Sequences : Retire(state, state', rank, seq))
    \/ (\E f \in Frames : PublishIds(state, state', f))
    \/ Hold(state, state')
    \/ BeginReal(state, state')
    \/ CompleteReal(state, state')
    \/ BeginAdvisory(state, state')
    \/ CompleteAdvisory(state, state')
    \/ WakeAndTake(state, state')
    \/ OfferAdvisory(state, state')
    \/ DropAdvisory(state, state')
    \/ Park(state, state')
    \/ WakeByPoll(state, state')
    \/ CloseReal(state, state')
    \/ CloseAdvisory(state, state')
    \/ Exit(state, state')
    \/ UNCHANGED state

\* At most one job per frame and generation may exist at a time.
DistinctJobs(s) ==
    \A i \in 1..Len(s), j \in 1..Len(s) :
        i # j => <<s[i].frame, s[i].gen>> # <<s[j].frame, s[j].gen>>

TypeOK ==
    /\ DOMAIN state.stateSeq = Frames
    /\ \A f \in Frames :
        /\ state.stateSeq[f] \in 0..MaxSequence
        /\ state.generation[f] \in 0..(RankCount * MaxSequence)
        /\ state.submitted[f] \subseteq Ranks
        /\ state.completed[f] \subseteq Ranks
        /\ state.waiting[f] \subseteq Ranks
        /\ state.rowCount[f] \in 0..MaxRows
        /\ state.published[f] \in 0..(RankCount * MaxSequence)
        /\ state.idsReady[f] \in 0..(RankCount * MaxSequence)
    /\ Len(state.queue) <= RealCapacity
    /\ \A i \in 1..Len(state.queue) :
        /\ state.queue[i].frame \in Frames
        /\ state.queue[i].gen \in 1..(RankCount * MaxSequence)
    /\ state.job = NoJob
         \/ \E f \in Frames, g \in 1..(RankCount * MaxSequence) :
              /\ state.job.frame = f
              /\ state.job.gen = g
    /\ state.held = NoJob
         \/ \E f \in Frames, g \in 1..(RankCount * MaxSequence) :
              /\ state.held.frame = f
              /\ state.held.gen = g
    /\ state.worker \in {Idle, Real, Advisory}
    /\ state.advisoryPending \in 0..AdvisoryCapacity
    /\ state.offers \in 0..MaxOffers
    /\ state.parked \in BOOLEAN
    /\ state.entryBacklog \in BOOLEAN
    /\ state.exited \in BOOLEAN
    /\ state.realClosed \in BOOLEAN
    /\ state.advisoryClosed \in BOOLEAN

\* Nobody retires before the reader thread published the rows. A rank may hold an
\* unreleased wait (that is the normal state right after a submission), so the
\* wait set is constrained by WaitingRanksSubmitted instead: what must never
\* happen is a retirement, which follows the transfer event, without publication.
RetirementRequiresPublication ==
    \A f \in Frames :
        state.completed[f] # {} =>
            /\ state.published[f] = state.generation[f]
            /\ state.published[f] # 0

\* Submission bookkeeping stays nested: retired and waiting ranks are subsets of
\* the ranks that submitted for the current occupant.
CompletedRanksSubmitted ==
    \A f \in Frames : state.completed[f] \subseteq state.submitted[f]

WaitingRanksSubmitted ==
    \A f \in Frames : state.waiting[f] \subseteq state.submitted[f]

\* Exactly one gather job exists per frame generation, in the queue or in hand:
\* a repeated submission must not enqueue a second gather, and a generation must
\* not be gathered twice. Generations are per frame, so the identity is the pair.
OneJobPerGeneration ==
    /\ DistinctJobs(state.queue)
    /\ (state.job # NoJob => \A i \in 1..Len(state.queue) :
            <<state.job.frame, state.job.gen>>
                # <<state.queue[i].frame, state.queue[i].gen>>)
    /\ (state.held # NoJob => \A i \in 1..Len(state.queue) :
            <<state.held.frame, state.held.gen>>
                # <<state.queue[i].frame, state.queue[i].gen>>)
    /\ (state.held # NoJob /\ state.job # NoJob =>
            <<state.held.frame, state.held.gen>> # <<state.job.frame, state.job.gen>>)

\* A queued, held, or in-flight job always names the generation its frame holds,
\* so a stale gather can never publish into a reused frame.
InFlightMatchesGeneration ==
    /\ \A i \in 1..Len(state.queue) :
         state.queue[i].gen = state.generation[state.queue[i].frame]
    /\ state.job # NoJob => state.job.gen = state.generation[state.job.frame]
    /\ state.held # NoJob => state.held.gen = state.generation[state.held.frame]

\* A published value is always a generation that frame actually holds, so an
\* advisory batch cannot release a waiting graph.
PublishedIsCurrentGeneration ==
    \A f \in Frames : state.published[f] # 0 => state.published[f] = state.generation[f]

\* The reader thread never begins advisory work while a gather could start. The
\* entry record makes the decision checkable after a later submission or
\* publication.
NoAdvisoryUnderRealBacklog ==
    state.worker = Advisory => ~state.entryBacklog

\* Shutdown cannot strand real work: the thread exits only with an empty
\* submission channel, nothing held or in hand, and a closed submission sender.
ExitedOnlyWhenDrained ==
    state.exited =>
        /\ Len(state.queue) = 0
        /\ state.worker = Idle
        /\ state.held = NoJob
        /\ state.job = NoJob
        /\ state.realClosed

\* An advisory batch never holds a gather job, and therefore never writes a
\* frame's staging buffer or publishes its generation: job is set only by
\* BeginReal. The frame bookkeeping itself changes only on Submit.
AdvisoryNeverOwnsFrame ==
    state.worker = Advisory => state.job = NoJob

\* Liveness. With bounded sequences and offers, and fair worker actions, every
\* queued gather is published and every accepted advisory batch is handled.
Drained == Len(state.queue) = 0 /\ state.worker = Idle /\ state.held = NoJob
           /\ state.job = NoJob
AdvisoryHandled == state.advisoryPending = 0
ProgressActions ==
    \/ Hold(state, state')
    \/ BeginReal(state, state')
    \/ CompleteReal(state, state')
    \/ BeginAdvisory(state, state')
    \/ CompleteAdvisory(state, state')
    \/ WakeAndTake(state, state')

\* Nothing is stranded while the reader thread is alive: every queued gather is
\* published, the thread settles, every accepted advisory batch is either warmed
\* or dropped by a shutdown, and a thread whose channels are closed is eventually
\* joined. Abandoning queued advisory work at shutdown is allowed: `Layer::Drop`
\* cancels the table before joining, so that work would have been discarded anyway.
Liveness ==
    /\ <>[]Drained
    /\ <>[](state.exited \/ AdvisoryHandled)
    /\ (state.realClosed /\ state.advisoryClosed => <> state.exited)

\* A launched graph eventually runs and publishes its ids: without that
\* environment assumption a held gather could wait forever.
Spec == Init /\ [][Next]_state /\ WF_state(ProgressActions)
             /\ WF_state(Exit(state, state'))
             /\ \A f \in Frames : WF_state(PublishIds(state, state', f))

=============================================================================
