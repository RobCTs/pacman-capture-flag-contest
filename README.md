# pacman-capture-flag-contest

1 Introduction
The Eutopia Pacman contest is an activity consisting of a multiplayer capture-the-
ag variant of Pacman,
where agents control both Pacman and ghosts in coordinated team-based strategies. Students from dif-
ferent EUTOPIA universities compete with each other through their programmed agents. Currently both
University of Ljubljana and Universitat Pompeu Fabra (UPF) are participating organizations. UPF is also
the tournament organizer, which hosts and run the tournaments in the HDTIC cluster1.
The project is based on the material from the CS188 course Introduction to Articial Intelligence at
Berkeley2, which was extended for the AI course in 2017 by lecturer Prof. Sebastian Sardina at the Royal
Melbourne Institute of Technology (RMIT University) and Dr. Nir Lipovetzky at University of Melbourne
(UoM)3. UPF has refactored the RMIT and UoM code. All the source code is written in Python.

2 Rules of Pacman Capture the Flag
2.1 Layout
The Pacman map is now divided into two halves: blue (right) and red (left). Red agents (which all have
even indices) must defend the red food while trying to eat the blue food. When on the red side, a red
agent is a ghost. When crossing into enemy territory, the agent becomes a Pacman.
2.2 Scoring
As a Pacman eats food dots, those food dots are stored up inside of that Pacman and removed from the
board. When a Pacman returns to his side of the board, he \deposits" the food dots he is carrying, earning
one point per food pellet delivered. Red team scores are positive, while Blue team scores are negative.
If Pacman is eaten by a ghost before reaching his own side of the board, he will explode into a cloud of
food dots that will be deposited back onto the board.
2.3 Eating Pacman
When a Pacman is eaten by an opposing ghost, the Pacman returns to its starting position (as a ghost).
No points are awarded for eating an opponent.
2.4 Power Capsules
If Pacman eats a power capsule, agents on the opposing team become \scared" for the next 40 moves,
or until they are eaten and respawn, whichever comes sooner. Agents that are \scared" are susceptible
while in the form of ghosts (i.e. while on their own team's side) to being eaten by Pacman. Specically,
if Pacman collides with a \scared" ghost, Pacman is unaected and the ghost respawns at its starting
position (no longer in the \scared" state).
2.5 Observations
Agents can only observe an opponent's conguration (position and direction) if they or their teammate is
within 5 squares (Manhattan distance). In addition, an agent always gets a noisy distance reading for each
agent on the board, which can be used to approximately locate unobserved opponents.
2.6 Winning
A game ends when one team returns all but two of the opponents' dots. Games are also limited to 1200
agent moves (300 moves per each of the four agents). If this move limit is reached, whichever team has
returned the most food wins. If the score is zero (i.e., tied) this is recorded as a tie game.
2.7 Computation Time
We will run your submissions on the UPF cluster, SNOW. Tournaments will generate many processes that
have to be executed without overloading the system. Therefore, each agent has 1 second to return each
action. Each move which does not return within one second will incur a warning. After three warnings, or
any single move taking more than 3 seconds, the game is forfeit. There will be an initial start-up allowance
of 15 seconds (use the registerInitialState function). If your agent times out or otherwise throws an
exception, an error message will be present in the log les, which you can download from the results page.
