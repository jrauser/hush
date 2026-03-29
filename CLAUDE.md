** Communicate **

If you're not sure about something ask me.  Don't be afraid to push back if you think I'm on the wrong track.

** Be a professional **

When I was an SDE, the thing I wanted most was the respect of my peers.  This meant having good software engineering hygiene and judgement.  This includes, at least:
  * Good variable/function/class names, no magic constants.
  * Code is readable.  Functions are kept short.
  * An appropriate amount unit tests, certainly more than zero; err on the side of too much testing
  * Don't repeat yourself (DRY)
  * Separation of concerns
  * Ya ain't gonna need it (YAGNI)
  * Keep it simple stupid (KISS).  
  * Compiler warnings turned up high and warnings not ignored
  * Just the right amount of logging and/or telemetry.  Consider how the system will behave when things go wrong and how other humans will diagnose and debug it.
 
  I'd like you to strive for respect in the same way. When people read your design documents and code they think "This was written by a real pro.  I can trust this person's judgement."  Being a pro does not mean slathering on every so-called best practice in the book.  If a project can be kept simple it should be kept simple.

** Documentation **

Below is the plan for a normal engagement.  If we discover major new requirements along the way we will have to improvise.

* I will write an `initial_requirements.md` file with my conception of the project. Agents should not edit this file.  If new requirements are discovered via planning I'll update this file.
* You will write a `plan.md` file.  We'll work together to hone the plan.  You're responsible for keeping plan.md up to date, but once we start implementation it should remain largely static. Updates should be marked as such and should be small.  More substantial learnings can go in `implementation_notes.md` below.
* During implementation, document deviations from the plan, surprising technical constraints, and non-obvious design decisions in `implementation_notes.md`.
* At the end I may ask you to write a `postmortem.md` with a summary of everything we did and learned through the course of the project.
* After changing behavior, consider impact on README.md and any other documentation.

** Testing **

* After any work, verify that tests pass.  Run ruff and fix any complaints.