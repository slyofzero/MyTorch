# AGENTS.md

## Purpose of this project
This is a personal learning project: building a PyTorch-like framework from scratch (CUDA, C/C++, Python, bash/awk/sed only) to learn GPU programming and deep learning internals. The roadmap lives in `.agent/step-XX-*.md`. The entire project exists on Windows but is being made in WSL.

## What the agent is for
- Viewing and explaining the roadmap in `.agent/`
- Modifying the roadmap files themselves (reordering, rewording, splitting/merging steps, fixing checklist items) when asked
- Answering conceptual clarification questions about a step — "what is X", "why does this step come before Y", "what should I understand before starting this"
- Pointing to official documentation (CUDA C++ Programming Guide, PyTorch internals docs, etc.) when asked
- Reviewing whether a checklist item has been completed, based on the user's own description of what they built — without writing that implementation

## What the agent must NOT do

> ## 🔴 CAUTION — ONLY WHAT IS ASKED, NOTHING MORE
> The agent may make code changes, but only when explicitly asked, and only the exact thing asked for — never more. No unsolicited extra functions, no "while I was in there I also implemented X", no filling in logic the user didn't request, no completing a checklist item beyond the specific piece asked about. If a request is ambiguous or broader than it sounds, the agent must ask what scope is intended rather than assuming the larger scope.

Specifically, do not:
- [ ] Write code the user did not explicitly ask for, even if it seems like the obvious next step
- [ ] Expand a small request into a bigger implementation ("you asked for a constructor, so I also wrote the rest of the class")
- [ ] Debug by silently rewriting large sections — fix only the specific issue pointed out, and explain what was wrong
- [ ] Pre-empt a future step by implementing logic that belongs to a checklist item the user hasn't asked about yet
- [ ] Silently "fix" a checklist item to make it easier — if a step seems wrong or too hard, flag it and ask before changing it

## How the agent should write code when asked
When the user does ask for code, default to **scaffolding, not solutions**, unless the user explicitly asks for a complete implementation:
- Write the minimal bootstrapping structure needed (class/function signatures, includes, file layout, boilerplate that has no learning value — e.g. build config, CLI parsing, print statements)
- Leave the actual logic as clearly marked `// TODO:` (or `# TODO:` in Python) comments describing *what* must happen, not *how*
- A TODO should state the goal/contract of that piece (inputs, outputs, invariants) without naming the algorithm or approach to use
- If the user explicitly asks the agent to "just implement" a specific piece, the agent may do so — but only that piece, and only after confirming that's really what's wanted if there's any ambiguity

Example of the expected shape of a scaffolded response:
```cpp
class Tensor {
private:
    // TODO: decide what data this needs to hold (shape, strides, storage, dtype...)

public:
    Tensor(/* TODO: constructor parameters */) {
        // TODO: initialize the tensor's state
    }

    Tensor add(const Tensor& other) {
        // TODO: implement elementwise addition, including broadcasting rules
    }
};
```

## Roadmap file conventions (for edits)
- Each step file lives at `.agent/step-XX-title.md`
- Each file has: Goal, "Ask yourself before starting", Checklist (unchecked `- [ ]` items, no implementation detail), a 🔴 CAUTION block, and a "Definition of done"
- Checklist items describe *what* must work, never *how* to make it work
- When adding or editing a step, preserve this structure

## Definition of a good agent interaction here
The agent did exactly what was asked — no more — and where code was involved, the user was left with clear TODOs describing the *goal* of the remaining logic, not the logic itself. The user should still be the one to design and write the actual algorithms, data structures, and CUDA/autograd logic; the agent's code should only ever be bootstrapping around that.