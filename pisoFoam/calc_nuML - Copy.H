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
// standardize features

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
	
	/////////////
/*
forAll(mesh.C(), i)
{

    X[i][0]  = d[i];
    X[i][1]  = F1[i];
    X[i][2]  = F2[i];
    X[i][3]  = F3[i];
    X[i][4]  = F4[i];
    X[i][5]  = F5[i];

//    X[i][0]  = F6[i];

    X[i][6]  = F7[i];
    X[i][7]  = F10[i];
    X[i][8]  = F11[i];

	X[i][9]   = Q1.component(0)[i]; //symmTensor xx
    X[i][10]  = Q1.component(1)[i]; //symmTensor xy
    X[i][11]  = Q1.component(3)[i]; //symmTensor yx	
	X[i][12]  = Q2.component(1)[i]; //asymmTensor xy	
    X[i][13]  = Q3.component(0)[i]; //vector x
    X[i][14]  = Q3.component(1)[i]; //vector y	
	X[i][15]  = Q4.component(0)[i]; //vector x
    X[i][16]  = Q4.component(1)[i]; //vector y		
    X[i][17]  = Q5[i]; //vector y

    //X[i][10] = I1[i];
    //X[i][11] = I2[i];
    //X[i][12] = I3[i];
    //X[i][13] = I4[i];
    //X[i][14] = I5[i];
}

label start = mesh.C().size();
forAll(mesh.boundaryMesh(), ipatch)
{
    if(mesh.boundaryMesh().types()[ipatch] == "empty")
        continue;

    forAll(mesh.boundaryMesh()[ipatch].faceCentres(), iface)
    {

    	X[start + iface][0] = d.boundaryField()[ipatch][iface];
        X[start + iface][1] = F1.boundaryField()[ipatch][iface];
        X[start + iface][2] = F2.boundaryField()[ipatch][iface];
        X[start + iface][3] = F3.boundaryField()[ipatch][iface];
        X[start + iface][4] = F4.boundaryField()[ipatch][iface];
        X[start + iface][5] = F5.boundaryField()[ipatch][iface];

//        X[start + iface][0] = F6.boundaryField()[ipatch][iface];

        X[start + iface][6] = F7.boundaryField()[ipatch][iface];
        X[start + iface][7] = F10.boundaryField()[ipatch][iface];
        X[start + iface][8] = F11.boundaryField()[ipatch][iface];

		X[start + iface][9]   = Q1.component(0)[i]; //symmTensor xx
		X[start + iface][10]  = Q1.component(1)[i]; //symmTensor xy
		X[start + iface][11]  = Q1.component(3)[i]; //symmTensor yx	
		X[start + iface][12]  = Q2.component(1)[i]; //asymmTensor xy	
		X[start + iface][13]  = Q3.component(0)[i]; //vector x
		X[start + iface][14]  = Q3.component(1)[i]; //vector y	
		X[start + iface][15]  = Q4.component(0)[i]; //vector x
		X[start + iface][16]  = Q4.component(1)[i]; //vector y		
		X[start + iface][17]  = Q5[i]; //vector y
    }
    start += mesh.boundaryMesh()[ipatch].faceCentres().size();
}
*/
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
	
	/*
    //multiplicatoin
    RectangularMatrix <doubleScalar> net1 = Xnorm * W1;
    //RectangularMatrix <doubleScalar> net1 = Xtest5 * W1;
    // addition
    for(int i = 0; i < net1.n(); ++i)
    {
        for(int j = 0; j < net1.m(); ++j)
        {
            net1[i][j] += b1[j][0];
        }
    }

    // batch normalizaiton
    for(int i = 0; i < net1.n(); ++i)
    {
        for(int j = 0; j < net1.m(); ++j)
        {
			net1[i][j] -= running_mean1[j][0];
            net1[i][j] /= Foam::sqrt(running_var1[j][0] + eps_bn);
            net1[i][j] *= gamma1[j][0];
            net1[i][j] += beta1 [j][0];
        }
    }

    // relu
    for(int i = 0; i < net1.n(); ++i)
    {
        for(int j = 0; j < net1.m(); ++j)
        {
            if(net1[i][j] < 0)
                net1[i][j] = 0;
        }   
    } 
	*/
    Info << "~~~~~~~~~~~~~~~~~~2nd layer~~~~~~~~~~~~~~~~~~" << endl;
	
	RectangularMatrix <doubleScalar> net2 = net1 * W2;
	add_b(net2, b2);
	batch_norm(net2, running_mean2, running_var2, gamma2,beta2,eps_bn);
	relu(net2);
	
	/*
    //multiplicatoin
    RectangularMatrix <doubleScalar> net2 = net1 * W2;
    // addition
    for(int i = 0; i < net2.n(); ++i)
    {
        for(int j = 0; j < net2.m(); ++j)
        {
            net2[i][j] += b2[j][0];
        }
    }

    // batch normalizaiton
    for(int i = 0; i < net2.n(); ++i)
    {
        for(int j = 0; j < net2.m(); ++j)
        {
            net2[i][j] -= running_mean2[j][0];
            net2[i][j] /= Foam::sqrt(running_var2[j][0] + eps_bn);
            net2[i][j] *= gamma2[j][0];
            net2[i][j] += beta2 [j][0];
        }
    }

    // relu
    for(int i = 0; i < net2.n(); ++i)
    {
        for(int j = 0; j < net2.m(); ++j)
        {
            if(net2[i][j] < 0)
                net2[i][j] = 0;
        }
    }
	*/
	Info << "net2" << net2 << endl;

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

    Info << "~~~~~~~~~~~~~~~~~~last layer~~~~~~~~~~~~~~~~~~" << endl;

	RectangularMatrix <doubleScalar> net5 = net4 * W5;
	add_b(net5, b5);
	
	/*
    //multiplicatoin
    RectangularMatrix <doubleScalar> net5 = net4 * W5;
    // addition
    for(int i = 0; i < net5.n(); ++i)
    {
        for(int j = 0; j < net5.m(); ++j)
        {
            net5[i][j] += b5[j][0];
        }
    }
    */
    Y = net5;
	Info << "Y" << Y << endl;
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
//Y =  10. * Y;

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


