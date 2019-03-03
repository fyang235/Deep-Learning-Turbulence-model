/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2012 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------

					this file is for variables updating

\*---------------------------------------------------------------------------*/
//update varibels for features, april 17 2018
//note: p is actually p/rho
Info << "Updating varibels for features\n" << endl;
	tao = turbulence->R()/1.;
	S   = symm(fvc::grad(U));
	R   = skew(fvc::grad(U));
	k_y  = turbulence->k()/1.;
	epsilon_ = turbulence->epsilon()/1.;
	nuLES    = (-1*(tao && S) + 2/3.*k_y*tr(S))/Foam::max(2*(S && S), eps1);
	S_ = k_y/Foam::max(eps2, epsilon_)*S;
	R_ = k_y/Foam::max(eps2, epsilon_)*R;
//------------------------------------------------------------------------------
// update features

Info << "Updating I features" << endl;
	I1 = tr(S_&S_);
	I2 = tr(S_&S_&S_);
	I3 = tr(R_&R_);
	I4 = tr(R_&R_&S_);
	I5 = tr(R_&R_&S_&S_);
	I6 = tr(R_&R_&S_&R_&S_&S_);

//------------------------------------------------------------------------------
//d  = getDistance2Walls(mesh); d is contant
//features from J. Ling, and J. Templeton https://doi.org/10.1063/1.4927765
Info << "Updating d and F features\n" << endl;
	F1 =  ((R&&R) - (S&&S)) /Foam::max((R&&R) + (S&&S), eps3);
	F2 = k_y /Foam::max( (0.5 * Foam::magSqr(U) + k_y), eps4);
	F3 = Foam::min( (Foam::sqrt(k_y)*d) / (50*turbulence->nu()), 2. );

	numerator_p = U & fvc::grad(p);
	sqrDpdx = Foam::magSqr(fvc::grad(p));
	sqrU    = Foam::magSqr(U);
	F4 = numerator_p / Foam::max(( Foam::sqrt(sqrDpdx*sqrU) + Foam::mag(numerator_p) ), eps5);

	F5 = (Foam::sqrt(S&&S) * k_y) /Foam::max( (Foam::sqrt(S&&S) * k_y + epsilon_), eps6 );
	F6 = turbulence->nut() / ( 100*turbulence->nu() + turbulence->nut() );

	magP = Foam::mag(fvc::grad(p));
	forAll(mesh.C(), i)
	{
		Usqr[i] = Foam::vector(sqr(U[i].component(vector::X)), sqr(U[i].component(vector::Y)), sqr(U[i].component(vector::Z)));
	}
	F7 = magP /Foam::max( magP + 0.5*rho*fvc::div(Usqr), eps7 );//!!!

	//scalar numerator_U = Foam::mag( (U*U) && (fvc::grad(U)*fvc::grad(U)) );
	//scalar nuderSqrt = 
	//F9 = 

	F10 = (U&fvc::grad(k_y)) /Foam::max( (Foam::mag(tao && S.T()) + (U&fvc::grad(k_y))), eps8 );
	F11 = Foam::sqrt(tao&&tao) /Foam::max( k_y + Foam::sqrt(tao&&tao), eps9 );
//------------------------------------------------------------------------------
Info << "Updating Q features\n" << endl;
	Q1 = S*k_y / Foam::max( Foam::sqrt(S&&S)*k_y + epsilon_, eps10 );
	Q2 = R / Foam::max( Foam::sqrt(R&&R)*2 , eps11 );
	Q3 = fvc::grad(p) / Foam::max( Foam::mag(fvc::grad(p)) + Foam::mag(fvc::ddt(U)+fvc::div(phi, U)), eps12 );
	Q4 = fvc::grad(k_y)*Foam::sqrt(k_y) / Foam::max( Foam::mag(fvc::grad(k_y))*Foam::sqrt(k_y) + epsilon_, eps13 );
	Q5 = k_y / Foam::max( k_y + turbulence->nu()*Foam::sqrt(S&&S), eps14 );


