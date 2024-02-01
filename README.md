# Using Deep Learning to Referee Fencing
<div style="text-align: justify"> 
In fencing, when both competitors hit eachother within a few miliseconds, both sides of the scoreboard will light up. Then the referee has to make a decision on who's point it is. As a rough rule of thumb, it goes to who took more initiative or who had more control over what happened. So I've experimented with training a model to fill in for the referee! So far, it has an accuracy of ~70% on the test set, up from the 35% that random chance would give (choosing between left, right and no point eitherway).

<h3> Building the Dataset </h3>
Fencing game continues till one of the players reaches score of 15. 
We break each video game into clips of plays:
- We identify each play ending by the light in one/or both sides of the scoreboard. 
- The clip consists of 50 frames from this ending point backward. 
- The label for the play, was taken from the next frames, by the diffrence in the score between the previous and next play.
- The score number was identified by finding maximum correlation between the fixed placed number, and pre-taken templates of numbers.
- We kept clips where both lights were on, meaning now it is the referee decision of who won the play. 

<h3> Key Descriptors for optimizing the problem </h3>

Pose Estimation Advantages:
- Data is truly unbiased by player country, name, current score, etc..
- The network can focus on relevant data, filtering unnecessary and biased information
Filtering just the fencing players from the scene
- Pose information seems to us very relevant to the play

Optical Flow Advantages:
- Focus on the movement changes, critical information for understanding who is the initiative of the play

<h3> Network </h3>
<p>
We used C3D where optical flow information and poses information where concatenated in channel dimension, and inserted together to the network as input
Also experimented with gathering the two players information in same channel vs in two different channels.
<p>
  <p>Hi there!

My name's Vance, I'm starting some related work trying to make a AI saber fencing referee. I'm really focused on in the box actions (attack, attack on prep, together) and the distinctions there. I'm exploring pose estimation. I'm working with CyrusOfChaos (Andrew) on this.
Would love to connect to learn about the work you've already done here!
Appreciate any help & guidance you can offer.

Best,
Vance</p>
<br>
  
<b>Pose Estimation</b>
<p align="center">
  <img src="LoYM4N80iEI-18-L-9766.gif?raw=true" alt=""/>
</p>

<b>Accurate Optical Flow from Pose Estimation</b>
<p align="center">
  <img src="optical_flow.gif?raw=true" alt=""/>
</p>

</div>

 
