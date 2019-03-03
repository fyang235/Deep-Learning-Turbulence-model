/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2012 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------

					this file is for defining varivles

\*---------------------------------------------------------------------------*/
//define varibales for ML features, march 25 2018
Info << "define varibales for ML features" << endl;
    Info<< "Creating field tao\n" << endl;
    volSymmTensorField tao
    (
        IOobject
        (
            "tao",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
	//mesh
        turbulence->R()
    );

    Info<< "Creating field S\n" << endl;
    volSymmTensorField S
    (
        IOobject
        (
            "S",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
	//mesh
        symm(fvc::grad(U))
    );

    Info<< "Creating field R\n" << endl;
    volTensorField R
    (
        IOobject
        (
            "R",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
	//mesh
        skew(fvc::grad(U))
    );


    Info<< "Creating field k_y\n" << endl;
    volScalarField k_y
    (
        IOobject
        (
            "k_y",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
	//mesh
        turbulence->k()
    );

    Info<< "Creating field epsilon_\n" << endl;
    volScalarField epsilon_
    (
        IOobject
        (
            "epsilon_",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
	//mesh
        turbulence->epsilon()
    );
	const scalar eps = 1e-30;
	const dimensionedScalar eps1 ("eps1", dimensionSet(0, 0, -2, 0, 0, 0, 0),eps);

    Info<< "Creating field nuLES\n" << endl;
    volScalarField nuLES
    (
        IOobject
        (
            "nuLES",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
	//mesh
        (-1*(tao && S) + 2/3.*k_y*tr(S))/Foam::max(2*(S && S), eps1)
    );

	const dimensionedScalar eps2 ("eps2", dimensionSet(0, 2, -3, 0, 0, 0, 0),eps);

    Info<< "Creating field S_less\n" << endl;
    volSymmTensorField S_
    (
        IOobject
        (
            "S_",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
	//mesh
        k_y/Foam::max(eps2, epsilon_)*S
    );


    Info<< "Creating field R_less\n" << endl;
    volTensorField R_
    (
        IOobject
        (
            "R_",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
	//mesh
        k_y/Foam::max(eps2, epsilon_)*R
    );
//------------------------------------------------------------------------------
// define I features
Info << "define features" << endl;
    Info<< "Creating field I1\n" << endl;
    volScalarField I1
    (
        IOobject
        (
            "I1",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
	//mesh
    	tr(S_&S_)
    );

    Info<< "Creating field I2\n" << endl;
    volScalarField I2
    (
        IOobject
        (
            "I2",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
	//mesh
        tr(S_&S_&S_)
    );

    Info<< "Creating field I3\n" << endl;
    volScalarField I3
    (
        IOobject
        (
            "I3",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
	//mesh
        tr(R_&R_)
    );

    Info<< "Creating field I4\n" << endl;
    volScalarField I4
    (
        IOobject
        (
            "I4",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
	//mesh
        tr(R_&R_&S_)
    );

    Info<< "Creating field I5\n" << endl;
    volScalarField I5
    (
        IOobject
        (
            "I5",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
	//mesh
        tr(R_&R_&S_&S_)
    );

    Info<< "Creating field I6\n" << endl;
    volScalarField I6
    (
        IOobject
        (
            "I6",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
	//mesh
        tr(R_&R_&S_&R_&S_&S_)
    );

// define d and F features
    volScalarField d
    (
        IOobject
        (
            "d",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        //mesh,
	getDistance2Walls(mesh)/characteristicDistance 
	//dimensionedScalar("minusOne", dimensionSet(0,1,0,0,0,0,0), 1e+10)
    );
//features from J. Ling, and J. Templeton https://doi.org/10.1063/1.4927765
	const dimensionedScalar eps3 ("eps3", dimensionSet(0, 0, -2, 0, 0, 0, 0),eps);
    Info<< "Creating field F1\n" << endl;
    volScalarField F1
    (
        IOobject
        (
            "F1",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        //mesh
        ((R&&R) - (S&&S)) /Foam::max((R&&R) + (S&&S), eps3)
    );
	const dimensionedScalar eps4 ("eps4", dimensionSet(0, 2, -2, 0, 0, 0, 0),eps);
    Info<< "Creating field F2\n" << endl;
    volScalarField F2
    (
        IOobject
        (
            "F2",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        //mesh
        k_y /Foam::max( (0.5 * Foam::magSqr(U) + k_y), eps4)
    );

    Info<< "Creating field F3\n" << endl;
    volScalarField F3
    (
        IOobject
        (
            "F3",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        //mesh
        Foam::min( (Foam::sqrt(k_y)*d) / (50*turbulence->nu()), 2. )
    );

	const dimensionedScalar eps5 ("eps5", dimensionSet(0, 2, -3, 0, 0, 0, 0),eps);
	volScalarField numerator_p = U & fvc::grad(p);
	volScalarField sqrDpdx = Foam::magSqr(fvc::grad(p));
	volScalarField sqrU    = Foam::magSqr(U);
    Info<< "Creating field F4\n" << endl;
    volScalarField F4
    (
        IOobject
        (
            "F4",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        //mesh
        numerator_p / Foam::max(( Foam::sqrt(sqrDpdx*sqrU) + Foam::mag(numerator_p) ), eps5)
    );
	const dimensionedScalar eps6 ("eps6", dimensionSet(0, 2, -3, 0, 0, 0, 0),eps);
    Info<< "Creating field F5\n" << endl;
    volScalarField F5
    (
        IOobject
        (
            "F5",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        //mesh
        (Foam::sqrt(S&&S) * k_y) /Foam::max( (Foam::sqrt(S&&S) * k_y + epsilon_), eps6 )
    );

    Info<< "Creating field F6\n" << endl;
    volScalarField F6
    (
        IOobject
        (
            "F6",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        //mesh
        turbulence->nut() / ( 100*turbulence->nu() + turbulence->nut() )
    );

	volScalarField magP = Foam::mag(fvc::grad(p));
    volScalarField rho//caution we did not find rho
    (
        IOobject
        (
            "rho",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
    	dimensionedScalar("rho", dimensionSet(0,0,0,0,0,0,0), 1.)//set dim to be zero
    );
	const dimensionedScalar eps7 ("eps7", dimensionSet(0, 1, -2, 0, 0, 0, 0),eps);

	volVectorField Usqr
    (
        IOobject
        (
            "Usqr",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedVector("Usqr", dimensionSet(0,2,-2,0,0,0,0), Foam::vector(0.,0.,0.))//set dim to be zero
    );
	forAll(mesh.C(), i)
	{
		Usqr[i] = Foam::vector(sqr(U[i].component(vector::X)), sqr(U[i].component(vector::Y)), sqr(U[i].component(vector::Z)));
	}
    Info<< "Creating field F7\n" << endl;
    volScalarField F7
    (
        IOobject
        (
            "F7",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        //mesh
        magP /Foam::max( magP + 0.5*rho*fvc::div(Usqr), eps7 )//!!!
    );
	const dimensionedScalar eps8 ("eps8", dimensionSet(0, 2, -3, 0, 0, 0, 0),eps);
    Info<< "Creating field F10\n" << endl;
    volScalarField F10
    (
        IOobject
        (
            "F10",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        //mesh
        (U&fvc::grad(k_y)) /Foam::max( (Foam::mag(tao && S.T()) + (U&fvc::grad(k_y))), eps8 )
    );
	const dimensionedScalar eps9 ("eps9", dimensionSet(0, 2, -2, 0, 0, 0, 0),eps);
    Info<< "Creating field F11\n" << endl;
    volScalarField F11
    (
        IOobject
        (
            "F11",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        //mesh
        Foam::sqrt(tao&&tao) /Foam::max( k_y + Foam::sqrt(tao&&tao), eps9 )
    );

// define Q features
	const dimensionedScalar eps10 ("eps10", dimensionSet(0, 2, -3, 0, 0, 0, 0),eps);
    Info<< "Creating field Q1\n" << endl;
    volSymmTensorField Q1
    (
        IOobject
        (
            "Q1",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        //mesh
        S*k_y / Foam::max( Foam::sqrt(S&&S)*k_y + epsilon_, eps10 )
    );

	const dimensionedScalar eps11 ("eps11", dimensionSet(0, 0, -1, 0, 0, 0, 0),eps);
    Info<< "Creating field Q2\n" << endl;
    volTensorField Q2
    (
        IOobject
        (
            "Q2",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        //mesh
        R / Foam::max( Foam::sqrt(R&&R)*2 , eps11 )
    );
	
	const dimensionedScalar eps12 ("eps12", dimensionSet(0, 1, -2, 0, 0, 0, 0),eps);
    Info<< "Creating field Q3\n" << endl;
    volVectorField Q3
    (
        IOobject
        (
            "Q3",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        //mesh
        fvc::grad(p) / Foam::max( Foam::mag(fvc::grad(p)) + Foam::mag(fvc::ddt(U)+fvc::div(phi, U)), eps12 ) // fvm ? fvc !!
    );

	const dimensionedScalar eps13 ("eps13", dimensionSet(0, 2, -3, 0, 0, 0, 0),eps);
    Info<< "Creating field Q4\n" << endl;
    volVectorField Q4
    (
        IOobject
        (
            "Q4",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        //mesh
        fvc::grad(k_y)*Foam::sqrt(k_y) / Foam::max( Foam::mag(fvc::grad(k_y))*Foam::sqrt(k_y) + epsilon_, eps13 ) 
    );	
	
	const dimensionedScalar eps14 ("eps14", dimensionSet(0, 2, -2, 0, 0, 0, 0),eps);
    Info<< "Creating field Q5\n" << endl;
    volScalarField Q5
    (
        IOobject
        (
            "Q5",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        //mesh
        k_y / Foam::max( k_y + turbulence->nu()*Foam::sqrt(S&&S), eps14 ) 
    );		
//------------------------------------------------------------------------------
	// define nuML
    Info<< "Creating field nuML\n" << endl;
    volScalarField nuML
    (
        IOobject
        (
            "nuML",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("minusOne", dimensionSet(0,2,-1,0,0,0,0), -1.)
    );

Info<< "\n after initialize nuML.size(): " << nuML.size() << endl;
//------------------------------------------------------------------------------
// read in ML parameters
Info << "Read in ML parameters\n" << endl;
RectangularMatrix<doubleScalar> mean(IFstream("MLparams/mean")());
RectangularMatrix<doubleScalar> std(IFstream("MLparams/std")());

RectangularMatrix<doubleScalar> W1(IFstream("MLparams/W1")());
RectangularMatrix<doubleScalar> W2(IFstream("MLparams/W2")());
RectangularMatrix<doubleScalar> W3(IFstream("MLparams/W3")());
RectangularMatrix<doubleScalar> b1(IFstream("MLparams/b1")());
RectangularMatrix<doubleScalar> b2(IFstream("MLparams/b2")());
RectangularMatrix<doubleScalar> b3(IFstream("MLparams/b3")());
RectangularMatrix<doubleScalar> beta1(IFstream("MLparams/beta1")());
RectangularMatrix<doubleScalar> beta2(IFstream("MLparams/beta2")());
RectangularMatrix<doubleScalar> gamma1(IFstream("MLparams/gamma1")());
RectangularMatrix<doubleScalar> gamma2(IFstream("MLparams/gamma2")());
RectangularMatrix<doubleScalar> running_mean1(IFstream("MLparams/running_mean1")());
RectangularMatrix<doubleScalar> running_mean2(IFstream("MLparams/running_mean2")());
RectangularMatrix<doubleScalar> running_var1(IFstream("MLparams/running_var1")());
RectangularMatrix<doubleScalar> running_var2(IFstream("MLparams/running_var2")());

// + 2 layers
/*
RectangularMatrix<doubleScalar> W4(IFstream("MLparams/W4")());
RectangularMatrix<doubleScalar> W5(IFstream("MLparams/W5")());
RectangularMatrix<doubleScalar> b4(IFstream("MLparams/b4")());
RectangularMatrix<doubleScalar> b5(IFstream("MLparams/b5")());
RectangularMatrix<doubleScalar> beta3(IFstream("MLparams/beta3")());
RectangularMatrix<doubleScalar> beta4(IFstream("MLparams/beta4")());
RectangularMatrix<doubleScalar> gamma3(IFstream("MLparams/gamma3")());
RectangularMatrix<doubleScalar> gamma4(IFstream("MLparams/gamma4")());
RectangularMatrix<doubleScalar> running_mean3(IFstream("MLparams/running_mean3")());
RectangularMatrix<doubleScalar> running_mean4(IFstream("MLparams/running_mean4")());
RectangularMatrix<doubleScalar> running_var3(IFstream("MLparams/running_var3")());
RectangularMatrix<doubleScalar> running_var4(IFstream("MLparams/running_var4")());
*/
// for test
//RectangularMatrix<doubleScalar> Xtest5(IFstream("MLparams/Xtest5.txt")());

//------------------------------------------------------------------------------
//concatenate features into a matrix
Info << "concatenate features into a matrix\n" << endl;
//label F  = 15;
const label F  = 18;
//const label H1 = 100;
//const label H2 = 100;
const label C  = 1;
label numOfInst = mesh.C().size();
forAll(mesh.boundaryMesh(), ipatch)
{
    if(mesh.boundaryMesh().types()[ipatch] == "empty")
	continue;

    numOfInst += mesh.boundaryMesh()[ipatch].faceCentres().size();
}
Info << "number of instance: " << numOfInst << endl;
RectangularMatrix <scalar>  X = RectangularMatrix <doubleScalar>( numOfInst, F, 0);
// Xtest5 is a 5 instance-case for debug
//RectangularMatrix <scalar>  X = Xtest5;
RectangularMatrix <scalar>  Y = RectangularMatrix <doubleScalar>( numOfInst, C, 0);

