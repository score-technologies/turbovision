# Private Track Scored Actions
This file is miner-facing and covers the actions currently scored on private track.

Source of truth for action names and scored set: `scorevision/utils/actions.py`.

Note: repo action `recovery` corresponds to the term `Loose Ball Recovery`.

| Action | Definition |
|---|---|
| `pass` | A pass is when a player kicks or throws the ball to one of their teammates. |
| `pass_received` | A Pass Received refers to the successful completion of a pass when a player gains control of the ball after it has been deliberately passed to them by a teammate. |
| `recovery` | A player gains possession after no team has possession of the ball or the ball is directed to them by an opponent. Active attempts to intercept the ball are excluded. |
| `tackle` | A player tries to stop an opposing player from progressing further with the ball or takes possession from an opposing player. |
| `interception` | A player intercepts an opposing team pass between two opposing players. |
| `ball_out_of_play` | The ball goes out of play. |
| `clearance` | A player clears the ball to safety by kick or header and eliminates immediate threat towards his/her own goal, regardless of who gains possession afterwards. |
| `take_on` | Situations in which a player in control of the ball moves past an opponent player. Awarded to the offensive player who performs the take-on. |
| `substitution` | Refers to the event when a player enters the match to replace a teammate. This occurs during a stoppage in play. |
| `block` | A player blocks a shot by an opposing player. |
| `aerial_duel` | An aerial duel occurs when two or more players attempt to gain possession of the ball in the air, typically using their head, for example after a long goal kick or a cross. At least one player must jump or clearly attempt to jump in order to contest the ball in the air. The key criterion is that the players are competing for the same ball, with physical contact or a visible attempt to win the ball. A separate event is recorded for each player involved in the duel. |
| `shot` | A Shot is an attempt made by a player to score a goal by striking or directing the ball towards the opponent's goal. |
| `save` | When the goalkeeper stops the ball from entering the net after a shot. |
| `foul` | Occurs when a player breaks the laws of the game through unfair play or actions such as tripping, pushing, or handling the ball, resulting in a free kick or penalty for the opposing team. Excluding offside events and advantages. Referee needs to stop play. |
| `goal` | To be awarded, the ball must pass completely over the goal line in the area between the posts and beneath the crossbar. Always comes with a shot event at the same time. |
