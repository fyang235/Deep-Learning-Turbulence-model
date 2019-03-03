if(MLtype == "training")
{
	Info << "writing trainging instances...\n" << endl;

	const label numOfFea = 25;
	//word featureArray[numOfFea] = {"d", "F1"}; 
	PtrList<volScalarField> featureList(numOfFea);
		featureList.set(0, d);
		featureList.set(1, F1);
		featureList.set(2, F2);
		featureList.set(3, F3);
		featureList.set(4, F4);
		featureList.set(5, F5);
		featureList.set(6, F6);
		featureList.set(7, F7);
		featureList.set(8, F10);
		featureList.set(9, F11);
			
		featureList.set(10, I1);
		featureList.set(11, I2);
		featureList.set(12, I3);
		featureList.set(13, I4);
		featureList.set(14, I5);
		
		featureList.set(15, Q1.component(0)); //symmTensor xx
		featureList.set(16, Q1.component(1)); //symmTensor xy
		featureList.set(17, Q1.component(3)); //symmTensor yx	
		featureList.set(18, Q2.component(1)); //asymmTensor xy	
		featureList.set(19, Q3.component(0)); //vector x
		featureList.set(20, Q3.component(1)); //vector y	
		featureList.set(21, Q4.component(0)); //vector x
		featureList.set(22, Q4.component(1)); //vector y		
		featureList.set(23, Q5); //vector y
	
		featureList.set(24, nuLES);   
	fileName outputFile("instances"+runTime.timeName());
	OFstream os(runTime.timeName()/outputFile);

	// print fea name
	os << fixed;
	forAll(featureList, i)
	{
		os << featureList[i].name() <<",\t\t\t\t";
	}
	os << endl;

	// print internal mesh value
	os << scientific << setprecision(5);
	forAll(mesh.C(), icell)
	{
		forAll(featureList, i)
		{
			os << featureList[i][icell] <<",\t\t";
		}    
		os << endl;
	}

	//print boundary mesh value
	forAll(mesh.boundaryMesh(), ipatch)
	{
		if(mesh.boundaryMesh().types()[ipatch] == "empty")
		continue;

		forAll(mesh.boundaryMesh()[ipatch], iface)
		{
		forAll(featureList, i)
			{
				os << featureList[i].boundaryField()[ipatch][iface] <<",\t\t";
			}
			os << endl;
		}
	}
}
/* don't write at testing to save time
if(MLtype == "testing" && moreInfo == true)
{
	//write X ------------------------------------------------------------------
	Info << "writing testing X...\n" << endl;
	fileName  Xfile("X"+runTime.timeName());
	OFstream osX(runTime.timeName()/Xfile);
	osX << fixed << scientific << setprecision(5);

	forAll(mesh.C(), i)
	{
		for(int j = 0; j < X.m(); ++j)
		osX << X[i][j] << ",\t";
		osX << endl;
	}

	label startWrite = mesh.C().size();
	forAll(mesh.boundaryMesh(), ipatch)
	{
		if(mesh.boundaryMesh().types()[ipatch] == "empty")
			continue;

		osX << "boundary " << mesh.boundaryMesh()[ipatch].name() << endl;
		forAll(mesh.boundaryMesh()[ipatch].faceCentres(), iface)
		{
		for(int j = 0; j < X.m(); ++j)
			osX << X[startWrite + iface][j] << ",\t";
		osX << endl;
		}
		startWrite += mesh.boundaryMesh()[ipatch].faceCentres().size();
	}


	//write Y ------------------------------------------------------------------
	Info << "writing testing Y...\n" << endl;
	fileName  Yfile("Y"+runTime.timeName());
	OFstream osY(runTime.timeName()/Yfile);
	osY << scientific << setprecision(5);

	forAll(mesh.C(), i)
	{
		osY << Y[i][0] << endl;
	}

	startWrite = mesh.C().size();
	forAll(mesh.boundaryMesh(), ipatch)
	{
		if(mesh.boundaryMesh().types()[ipatch] == "empty")
			continue;

		osY << "boundary " << mesh.boundaryMesh()[ipatch].name() << endl;
		forAll(mesh.boundaryMesh()[ipatch].faceCentres(), iface)
		{
			osY << Y[startWrite + iface][0] << endl;
		}
		startWrite += mesh.boundaryMesh()[ipatch].faceCentres().size();
	}
}
*/
