"""Action-class-aware streaming commit policy ("commit early, refine later").

The naive question "was the commit the gold intent?" is the wrong one for a live assistant:
saying "youtube ..." and having YouTube open before the sentence ends is the desired
behaviour, and "youtube play lofi" then REFINES that open into a search -- it does not
contradict it. What must never happen is an action that is wrong AND costly: typing into the
wrong window, closing the wrong app, playing the wrong thing.

Policy evaluated here, per prefix decision (pred, pmax):
  * 'none' on a prefix never commits (keep listening).
  * OPEN actions (open_notes/open_youtube/open_spotify/open_browser): commit at pmax>=tau
    with stability 1 -- cheap, reversible, and refinable.
  * MEDIA actions (pause/resume/next/volume): stability >= media_stable (default 2).
  * PLAY/SEARCH actions (play_youtube/play_spotify/play_liked_songs/search_web): stability
    >= play_stable (default 2); a prior OPEN of the same app is refined, not contradicted.
  * DEFERRED actions (type_text, close_app): never on a prefix -- they need the full text /
    the app name, so they execute at end of utterance from the final decision.
Refinement graph: open_youtube -> play_youtube; open_spotify -> play_spotify, play_liked_songs;
open_browser -> search_web; open_notes -> type_text (open then type).

Scoring, per actionable utterance: the SEQUENCE of executed actions (early commits in order,
then the end-of-utterance action if it differs) is compared with the accepted set
{gold, then, alt}: every executed action must be in the accepted set or a refinable ancestor
of one; otherwise the utterance counts as HARMFUL. Also reported: time to first action (word
index), fraction of utterances where the first action happened at or before the human commit
word, and false actions on out-of-scope utterances.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

OPEN = {"open_notes", "open_youtube", "open_spotify", "open_browser"}
MEDIA = {"pause_media", "resume_media", "next_track", "volume_up", "volume_down"}
PLAY = {"play_liked_songs"}                                   # no slot: may fire on a prefix
SLOT_PLAY = {"play_youtube", "play_spotify", "search_web"}   # need the full query: fire the OPEN ancestor now, the play at end
DEFERRED = {"type_text", "close_app"}                         # need full text / app name: end of utterance only
REFINES = {  # child -> ancestor OPEN action it implies
    "play_youtube": "open_youtube", "play_spotify": "open_spotify", "play_liked_songs": "open_spotify",
    "search_web": "open_browser", "type_text": "open_notes",
}


def load(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def executed_sequence(traj, tau, media_stable, play_stable, slot_stable=2):
    """Return list of (k, action) executed during streaming, then the final decision."""
    done = []
    run = []  # consecutive same-pred window
    for t in traj:
        a, p = t["pred"], t["pmax"]
        run = run + [a] if (run and run[-1] == a) else [a]
        if a == "none" or p < tau or a in DEFERRED:
            continue
        if a in SLOT_PLAY:
            # the query is not complete yet: open the app now (cheap, refinable), search at the end.
            # "play X" is platform-ambiguous until "on youtube/spotify" arrives, hence slot_stable.
            anc = REFINES[a]
            if len(run) >= slot_stable and anc not in [x for _, x in done]:
                done.append((t["k"], anc))
            continue
        need = 1 if a in OPEN else media_stable if a in MEDIA else play_stable
        if len(run) < need:
            continue
        if done and done[-1][1] == a:
            continue
        # refinement of a previous OPEN is allowed; a second OPEN of a different app is not skipped (it is an action)
        done.append((t["k"], a))
    final = traj[-1]["pred"]
    if final != "none" and (not done or done[-1][1] != final):
        done.append((traj[-1]["k"], final))  # end-of-utterance action (includes DEFERRED and SLOT_PLAY)
    return done


def consistent(action, accepted):
    return action in accepted or any(REFINES.get(x) == action for x in accepted)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    ap.add_argument("--full", required=True)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--media-stable", type=int, default=2)
    ap.add_argument("--play-stable", type=int, default=2)
    ap.add_argument("--slot-stable", type=int, default=2, help="consecutive partials before a slot-play prediction opens its app")
    ap.add_argument("--out")
    args = ap.parse_args()
    _exec = executed_sequence
    executed_sequence_local = lambda traj, tau, ms, ps: _exec(traj, tau, ms, ps, args.slot_stable)  # noqa: E731
    globals()["executed_sequence"] = executed_sequence_local
    S = load(args.stream)
    F = {r["id"]: r for r in load(args.full)}
    act = [s for s in S if s["gold"] != "none"]
    oos = [s for s in S if s["gold"] == "none"]

    harmful, first_k, before_human, n_actions, harm_rows, correct_final = 0, [], 0, [], [], 0
    for s in act:
        f = F[s["id"]]
        accepted = {s["gold"]} | ({f["then"]} if f.get("then") else set()) | ({f["alt"]} if f.get("alt") else set())
        seq = executed_sequence(s["trajectory"], args.tau, args.media_stable, args.play_stable)
        bad = [a for k, a in seq if not consistent(a, accepted)]
        if bad:
            harmful += 1
            harm_rows.append(dict(id=s["id"], style=s["style"], seq=seq, accepted=sorted(accepted), text=s["trajectory"][-1]["prefix"]))
        if seq:
            first_k.append(seq[0][0])
            if s["human_commit_word"] > 0 and seq[0][0] <= s["human_commit_word"]:
                before_human += 1
        n_actions.append(len(seq))
        correct_final += bool(seq) and consistent(seq[-1][1], accepted)
    oos_actions = [(s["id"], executed_sequence(s["trajectory"], args.tau, args.media_stable, args.play_stable)) for s in oos]
    oos_bad = [(i, q) for i, q in oos_actions if q]
    res = dict(
        policy=dict(tau=args.tau, media_stable=args.media_stable, play_stable=args.play_stable, slot_stable=args.slot_stable),
        actionable=len(act),
        harmful_rate=harmful / len(act), harmful=harmful,
        final_action_consistent_rate=correct_final / len(act),
        mean_first_action_word=statistics.mean(first_k) if first_k else None,
        first_action_at_or_before_human=before_human / len(act),
        mean_actions_per_utterance=statistics.mean(n_actions),
        oos_rows=len(oos), oos_false_actions=len(oos_bad), oos_false_action_rate=len(oos_bad) / len(oos) if oos else None,
        oos_false_detail=oos_bad, harmful_detail=harm_rows,
    )
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("detail")}, indent=2))
    for h in harm_rows:
        print("  HARM", h)
    for o in oos_bad:
        print("  OOS ", o)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
