The commands are broken into folders based on what project they are for.
Originally, all the commands were for the LLOWFS (32x32 camera).
There is also a `shared.md` in this folder and that contains commands that can be used regardless of the project.

**Note**: The VVC simulations can use either `cbm_vvc_mft` (which uses MFT) or `cbm_vvc_approx` (which does not use MFT).
In general, the two should produce roughly the same wavefronts.
However, for the base field and small aberrations, the two will produce noticabely different wavefronts.
Therefore, if a model is trained on data simulated with `cbm_vvc_approx`, then it should not be tested on small aberration wavefronts simulated with `cbm_vvc_mft`.

**Note**: This whole directory is a mess and commands are scattered throughout.
I have been trying to keep it semi-organized, but it could be much better.
You have been warned.
