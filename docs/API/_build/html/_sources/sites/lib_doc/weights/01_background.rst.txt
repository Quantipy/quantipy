.. toctree::
 	:maxdepth: 5
	:includehidden:

==========================
Background and methodology
==========================

``quantipy`` utilizes the *Rim* (sometimes also called *Raking*) weighting method,
an iterative fitting algorithm that tries to balance out multiple sample
frequencies simultaneously. It is rooted in the mathematical model developed in
the seminal academic paper by Deming/Stephan (1940) ([DeSt40]_). The following chapters
draw heavily from it.

The statistical problem
-----------------------

More often than not, market research professionals (and not only them!) are
required to weight their raw data collected via a survey to match a known
specific real-world distribution. This is the case when you try to weight your
sample to reflect the population distribution of a certain characteristic to
make it “representative” in one or more terms. Leaving unconsidered what a
“representative” sample actually is in the first place, let’s see what
“weighting data” comes down to and why weighting in order to achieve representativeness
can be quite a difficult task. Look at the following two examples:

1.  Your data contains an equal number of male and female respondents while in
the real world you know that women are a little bit more frequent than men.
In relative terms you have sampled 2 percentage points more men than women:

+-------+----------------+------------+---------------+
|       | Sample (N=100) | Population | Factors       |
+-------+----------------+------------+---------------+
| Men   | 50 %           | 48%        | 48/50 = 0.96  |
+-------+----------------+------------+---------------+
| Women | 50%            | 52%        | 52/50 = 1.04  |
+-------+----------------+------------+---------------+

That one is easy because you know each cell’s population frequencies and can
simply find the factors that will correct your sample to mirror the real-world
population. To weight you would simply compute the relevant factors by dividing
the desired population figure by the sample frequency and assign each case in
your data the respective result (based on his or her gender). The factors are coming from
your **one-dimensional** weighting matrix above.


2. You have a survey project that requires the sample to match the gender and age
distributions in real-world Germany and additionally should take into account
the distribution of iPad owners and the population frequencies of the federal
states.

Again, to weight the data you would need to calculate the cell ratios of target
vs. sample figures for the different sample characteristics. While you may be
able to find the **joint** distribution of age categories by gender, you will
have a hard time coming up e.g. with the correct figures for a **joint** distribution
of iPad owners per federal state by gender and age group.

To put it differently: You will not know the population’s cell target figures
for all weighting dimensions in all relevant cells of the **multi-dimensional**
weighting matrix. Since you need this information to assign each case a weight
factor to come up with the correct weighted distributions for the four sample
characteristics you would not be able to weight the data.
To illustrate the complexity of such a weighting scheme, the table below should
suit::


  ╔═════════╦═════════╦═══════════════════════╦═══════════════════════╦═════╗
  ║ State:  ║         ║ Bavaria               ║ Saxony                ║     ║
  ╠═════════╬═════════╬═══════╦═══════╦═══════╬═══════╦═══════╦═══════╬═════╣
  ║ Age:    ║         ║ 18-25 ║ 26-35 ║ 36-55 ║ 18-25 ║ 26-35 ║ 36-55 ║ ... ║
  ╠═════════╬═════════╬═══╦═══╬═══╦═══╬═══╦═══╬═══╦═══╬═══╦═══╬═══╦═══╬═════╣
  ║ Gender: ║         ║ m ║ f ║ m ║ f ║ m ║ f ║ m ║ f ║ m ║ f ║ m ║ f ║ ... ║
  ╠═════════╬═════════╬═══╬═══╬═══╬═══╬═══╬═══╬═══╬═══╬═══╬═══╬═══╬═══╬═════╣
  ║         ║ iPad    ║ ? ║ ? ║ ? ║ ? ║ ? ║ ? ║ ? ║ ? ║ ? ║ ? ║ ? ║ ? ║ ?   ║
  ╠═════════╬═════════╬═══╬═══╬═══╬═══╬═══╬═══╬═══╬═══╬═══╬═══╬═══╬═══╬═════╣
  ║         ║ no iPad ║ ? ║ ? ║ ? ║ ? ║ ? ║ ? ║ ? ║ ? ║ ? ║ ? ║ ? ║ ? ║ ?   ║
  ╚═════════╩═════════╩═══╩═══╩═══╩═══╩═══╩═══╩═══╩═══╩═══╩═══╩═══╩═══╩═════╝


Note that, you would also need to take into account the other joint distributions
of age by gender per federal state, iPad owners by age, and so on to get the
correct weight factors step by step: all cross-tabulation information for the
population that will not be available to you. Additionally, even if you would
have all the information necessary for your calculations, try to imagine the
amount of work that awaits to come up with the weight factors per cell
regarding getting all possible combinations right, then creating variables,
recoding those variables and then finally computing the ratios.

What is available regularly, however, is the distribution of people living in
Germany’s federal states and the distribution of iPad owners in general
(as per “Yes, have one,” “do not own one”), plus the age and gender frequencies.
This is where rim weighting comes into play.

Rim weighting concept
---------------------

Rim weighting in short can be described as an **iterative data fitting** process
that aims to apply a weight factor to each respondent’s case record in order to
match the target figures by altering the sample cell frequencies relevant to the
weighting matrix. Doing that, it will find the single cell’s ratios that are required
to come up with the correct targets per weight dimension – it will basically **estimate**
all the joint distribution information that is unknown.

The way this works can be summarized as follows: For each interlocking cell
coming from all categories of all the variables that are given to weight to, an
algorithm will compute the proportion necessary in a single specific cell that,
when summed over per column or respectively by row, will result in a column (row)
total per category that matches the target distribution. However, it will occur
that having balanced a column total to match, the row totals will be off.
This is where one iteration ends and another one begins starting now with the
weighted values from the previous run. This iterative process will continue
until a satisfying result in terms of an acceptable low amount of mismatch
between produced sample results and weight targets is reached.

In short: Simultaneous adjustment of all weight variables with the smallest
amount of data manipulation possible while forcing the maximum match between
sample and weight scheme.


.. rubric:: References

.. [DeSt40] Deming, W. Edwards; Stephan, Frederick F. (1940): On a Least Squares Adjustment of a Sampled Frequency Table When the Expected Marginal Totals are Known. In: Ann. Math. Statist. 11 , no. 4, pp. 427 - 444.
