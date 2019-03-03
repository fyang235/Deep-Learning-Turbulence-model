/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2012 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------

					calc machine learning nuML

\*---------------------------------------------------------------------------*/

//------------------------------------------------------------------------------

//word featureArray[numOfFea] = {"d", "F1"}; 
PtrList<volScalarField> featureList(F);
	featureList.set(0, d);
	featureList.set(1, F1);
	featureList.set(2, F2);
	featureList.set(3, F3);
	featureList.set(4, F4);
	featureList.set(5, F5);
//	featureList.set(6, F6);
	featureList.set(6, F7);
	featureList.set(7, F10);
	featureList.set(8, F11);
	
/*	
	featureList.set(10, I1);
	featureList.set(11, I2);
	featureList.set(12, I3);
	featureList.set(13, I4);
	featureList.set(14, I5);
*/
	
	featureList.set(9,  Q1.component(0)); //symmTensor xx
	featureList.set(10, Q1.component(1)); //symmTensor xy
	featureList.set(11, Q1.component(3)); //symmTensor yx	
	featureList.set(12, Q2.component(1)); //asymmTensor xy	
	featureList.set(13, Q3.component(0)); //vector x
	featureList.set(14, Q3.component(1)); //vector y	
	featureList.set(15, Q4.component(0)); //vector x
	featureList.set(16, Q4.component(1)); //vector y		
	featureList.set(17, Q5); //vector y
	
//	featureList.set(23, nuLES);   

// copy internal mesh value
forAll(mesh.C(), icell)
{
	forAll(featureList, i)
	{
		X[icell][i] = featureList[i][icell] ;
	}    
}

// copy boundary mesh value
label start = mesh.C().size();
forAll(mesh.boundaryMesh(), ipatch)
{
	if(mesh.boundaryMesh().types()[ipatch] == "empty")
	continue;

	forAll(mesh.boundaryMesh()[ipatch], iface)
	{
		forAll(featureList, i)
		{
			X[start + iface][i] = featureList[i].boundaryField()[ipatch][iface];
		}
	}
	start += mesh.boundaryMesh()[ipatch].faceCentres().size();
}
	
//------------------------------------------------------------------------------
// standardize features
Info << "standardize features\n" << endl;
RectangularMatrix <scalar>  Xnorm = X;
for(int i = 0; i < X.n(); ++i)
{
    for(int j = 0; j < X.m(); ++j)
    {
		Xnorm[i][j] -= mean[j][0];
		Xnorm[i][j] /= std [j][0];	
    }
}

//===============================start of Neural Network===============================
// forward pass
    scalar eps_bn = 1e-5;

    Info << "~~~~~~~~~~~~~~~~~~1st layer~~~~~~~~~~~~~~~~~~" << endl;

		RectangularMatrix <doubleScalar> net1 = Xnorm * W1;
		add_b(net1, b1);
		batch_norm(net1, running_mean1, running_var1, gamma1,beta1,eps_bn);
		relu(net1);
	
    Info << "~~~~~~~~~~~~~~~~~~2nd layer~~~~~~~~~~~~~~~~~~" << endl;
	
		RectangularMatrix <doubleScalar> net2 = net1 * W2;
		add_b(net2, b2);
		batch_norm(net2, running_mean2, running_var2, gamma2,beta2,eps_bn);
		relu(net2);
/*
    Info << "~~~~~~~~~~~~~~~~~~3rd layer~~~~~~~~~~~~~~~~~~" << endl;
	
		RectangularMatrix <doubleScalar> net3 = net2 * W3;
		add_b(net3, b3);
		batch_norm(net3, running_mean3, running_var3, gamma3,beta3,eps_bn);
		relu(net3);
	
	Info << "~~~~~~~~~~~~~~~~~~4th layer~~~~~~~~~~~~~~~~~~" << endl;

		RectangularMatrix <doubleScalar> net4 = net3 * W4;
		add_b(net4, b4);
		batch_norm(net4, running_mean4, running_var4, gamma4,beta4,eps_bn);
		relu(net4);
*/
    Info << "~~~~~~~~~~~~~~~~~~last layer~~~~~~~~~~~~~~~~~~" << endl;

		RectangularMatrix <doubleScalar> net3 = net2 * W3;
		add_b(net3, b3);
		Y = net3;
		
//=============================== end of Neural Network===============================
// denormalize label
Info << "denormalize label\n" << endl;
    for(int i = 0; i < Y.n(); ++i)
    {
        for(int j = 0; j < Y.m(); ++j)
        {
			// note the mean and std at the last position
            Y[i][j] *= std [F][0];
            Y[i][j] += mean[F][0];
        }
    }
//Info << "after denormalize" << Y <<endl;
//------------------------------------------------------------------------------
// calc nuML

//test x10
//Y =  0.000 * Y;

Info << "calc nuML\n" << endl;
forAll(mesh.C(), i)
{
    nuML[i] = Y[i][0];
}

start = mesh.C().size();
forAll(mesh.boundaryMesh(), ipatch)
{
    if(mesh.boundaryMesh().types()[ipatch] == "empty")
        continue;

    forAll(mesh.boundaryMesh()[ipatch].faceCentres(), iface)
    {
        nuML.boundaryField()[ipatch][iface] = Y[start + iface][0];
    }
    start += mesh.boundaryMesh()[ipatch].faceCentres().size();
}



//Info << "Y: "<< Y << endl;
//Info << "\n after calc nuML, nuML.size(): " << nuML.size() << endl;
//Info << nuML << endl;


