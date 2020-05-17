weak_ties_optimized contains the files that optimized loops.


in optimized folder:

diag_zero_filter; removes self loops from A, calculate A^2 and also creates a binary adj.matrix so if there is a tie weak 1 else 0 with A^2>0

out of diag_zero; calculates A^2 but then pass it to SGC so there migth be elements bigger than 1; not sure what happens with those elements but +/- I
refers to avoiding SGC from addingg self loop or not.


