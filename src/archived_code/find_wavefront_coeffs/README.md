These three scripts find the coefficients that make up a wavefront in different ways:

1. [V1] Projects the wavefront onto the propagated basis terms and solves the system of equations.
2. [V2] Minimizes the wavefront using the propagated basis terms.
3. [V3] Minimizes the wavefront by using actual forward model minimization.

Methods 1 and 2 do not work correctly due to the propagation through the optical setup being a nonlinear transformation.
Method 3 will work correctly, but it takes a very long time to run.

For more information on why methods 1 and 2 do not work, please refer to the Google Slides link:
https://docs.google.com/presentation/d/1ciZjWCaF4tTnLjTrsCuEpN1HK7io9aKQ-C1mNGXANno
