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
 
  I'd like you to strive for respect in the same way.  When people read your design documents and code they think "This was written by a real pro.  I can trust this person's judgement."  Being a pro does not mean slathering on every so-called best practice in the book.  If a project can be kept simple it should be kept simple.

** Documentation **

* `initial_requirements.md` is the sketch of v1 the system.  `v2_requirements.md` is the sketch of v2.  These were written by a human.  Do not edit these files.
* `plan_*.md` are agent authored plans.
* `implementation_notes.md` is an ongoing scratchpad of interesting discoveries or deviations from the plans.  Keep this file up to date as you implement.
* `TODO.md` is a scratchpad for bug reports and future directions.
* After changing behavior, consider impact on `README.md` and any other documentation.

** Testing **

* After any work, verify that tests pass.  Run ruff and fix any complaints.