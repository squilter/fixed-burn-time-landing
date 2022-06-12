# fixed-burn-time-landing

Generate an optimal control policy for throttling a model rocket motor to land a model rocket softly. The tricky part is that the motor burns for about 3.5 seconds and you can't stop it early.

This code sets up the problem as a discrete-time [dynamic programming](http://underactuated.mit.edu/dp.html) problem, and solves it through [backward induction](https://en.wikipedia.org/wiki/Backward_induction). The result of running the demo is a big lookup table, giving the ideal throttle based on the state of the vehicle, defined as:
- burn time remaining
- speed
- height

Run `pip3 install -r requirements.txt` first

Run `python3 demo.py` and it will generate a table and then show what the control policy would do with multiple starting conditions