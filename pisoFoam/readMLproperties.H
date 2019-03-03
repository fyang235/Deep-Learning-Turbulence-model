/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  2.2.2                                 |
|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/

Info << "Reading MLproperties.H ..." << endl;
	IOdictionary MLproperties
	(
		IOobject
		(
		"MLproperties",
		U.time().constant(),
		U.db(),
		IOobject :: MUST_READ,
		IOobject :: NO_WRITE
		)
	);


// read MLmode
	Switch MLmode(MLproperties.lookup("MLmode"));

	word   MLtype = "None";
	scalar characteristicDistance = 1.;
	Switch moreInfo = true;

	if(MLmode == true)
	{
		// read MLtype
		word tmpMLtype(MLproperties.lookup("MLtype"));
		MLtype = tmpMLtype;

		if(MLtype != "training" && MLtype != "testing")
		{
		Info << "Invalide MLtype! Please inpute training or testing."<< endl;
		}	

		// read characteristicDistance
		characteristicDistance = readScalar(MLproperties.lookup("characteristicDistance"));
		
		// read moreInfo
		Switch tmpMoreInfo = MLproperties.lookup("moreInfo");
		moreInfo = tmpMoreInfo;
	}

// print ML info
	Info << "\nML settings: \n" << endl;
	Info << "\t MLmode: " << MLmode << endl;
	if(MLmode == true)
	{
		Info << "\t MLtype: " << MLtype << endl;
		Info << "\t characteristicDistance: " << characteristicDistance << endl;
		Info << "\t moreInfo: " << moreInfo << endl;
	}
	Info << endl;








