ArCTIC python
=============

AlgoRithm for Charge Transfer Inefficiency (CTI) Correction
-----------------------------------------------------------

What is CTI?
------------


Correcting HST images
---------------------

Adding/removing CTI trails in images from any other camera
----------------------------------------------------------

1) Define a CCD

This specifies the way in which electrons fill the volume of a CCD pixel. 
Multiple phases within each pixel can be defining e.g. fraction_of_traps as a list.

2) Define readout electronics

This specifies the way in which electrons are moved between pixels. 
A default class exists, ROE(), for readout of standard imaging mode, assuming that electrons move instantaneously from (the first phase in) one pixel to another.
More complicated schemes exist to specify the movement of electrons between intermediate phases, readout of electronically injected charge, or trap ("pocket") pumping.

3) Define trap species

This specifies the 

Coming soon...





Jacob Kegerreis  
Richard Massey  
James Nightingale  

Formatted with [black](https://github.com/psf/black).
