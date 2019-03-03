/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2012 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------

					this file is for self defined functions

\*---------------------------------------------------------------------------*/
//-------------------------self defined fun-----------------------
scalar ymean(const scalarField& I)
{
    scalar sum = 0;
    for(int i = 0; i < I.size(); i++)
    {
        sum += I[i];
    }
    return sum/I.size();
}

scalar ystd(const scalarField& I)
{
    scalar squareSum = 0.;
    for(int i = 0; i < I.size(); i++)
    {
	squareSum += (I[i] - ymean(I))*(I[i] - ymean(I));
    }
    //sqrt(ymean((I - ymean(I))*(I - ymean(I)));
    return Foam::sqrt(squareSum/I.size());
}

scalarField standardize(const scalarField& I)
{
    return (I - ymean(I))/(ystd(I) + 1e-12);
}

RectangularMatrix<doubleScalar> sigmoid(const RectangularMatrix<doubleScalar>& A)
{
    RectangularMatrix<doubleScalar> A_(A.n(), A.m());
    //Info << "A.n(), A.m(): \n"<<A.n()<<'\n'<< A.m()<<endl;
    for(int i = 0; i < A.n(); i++)
    {
	for(int j = 0; j < A.m(); j++)
	{
	    A_[i][j] = 1./(1. + Foam::exp(-1*A[i][j]));
	    //Info << "A_[i][j]: " << A_[i][j] << endl;
	}
    }
    //Info << "A_: " << A_ << endl;
    return A_;
}

RectangularMatrix<doubleScalar> 
unstandardize (const RectangularMatrix<doubleScalar>& A, const scalar& Y_mean, const scalar& Y_std)
{
    RectangularMatrix<doubleScalar> A_(A.n(), A.m());
    for(int i = 0; i < A.n(); i++)
    {
        for(int j = 0; j < A.m(); j++)
        {
            A_[i][j] = A[i][j] * Y_std + Y_mean;
        }
    }
    return A_;
}

volScalarField getDistance2SingleWall(const List<vector>& bfaceCenter, const fvMesh& mesh)
{
    volScalarField distance
    (
        IOobject
        (
            "tmp",
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
	mesh,
        dimensionedScalar("minusOne", dimensionSet(0,1,0,0,0,0,0), 1e+11)
    );

    // calc internal mesh
    forAll(mesh.C(), icell)
    {
        scalar  minDistance(1e+9);
        forAll(bfaceCenter, iface)
        {
            scalar tmpDistance = Foam::mag(mesh.C()[icell] - bfaceCenter[iface]);
            minDistance = Foam::min(tmpDistance, minDistance);
        }
        distance[icell] = minDistance;
    }

    // calc boundary mesh
    //Info << "mesh.boundaryMesh()" << mesh.boundaryMesh() << endl;
    //Info << "distance.boundaryField()" << distance.boundaryField()<< endl;
    forAll(mesh.boundaryMesh(), ipatch)
    {   
	if(mesh.boundaryMesh().types()[ipatch] == "empty")
	    continue;

        forAll(mesh.boundaryMesh()[ipatch].faceCentres(), icell)
        {
            scalar  minDistance(1e+9);
            forAll(bfaceCenter, iface)
            {
                scalar tmpDistance = Foam::mag(mesh.boundaryMesh()[ipatch].faceCentres()[icell] - bfaceCenter[iface]);
                minDistance = Foam::min(tmpDistance, minDistance);
            }
            distance.boundaryField()[ipatch][icell] = minDistance;
        }
    }
    return distance;
}

volScalarField getDistance2Walls(const fvMesh& mesh)
{
    volScalarField distance
    (
        IOobject
        (
            "tmp",
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
	mesh,
        dimensionedScalar("minusOne", dimensionSet(0,1,0,0,0,0,0), 1e+10)
    );

    forAll(mesh.boundaryMesh(), ipatch)
    {
        word BCtype = mesh.boundaryMesh().types()[ipatch];
        if (BCtype == "wall")
        {
            const List<vector> bfaceCenter = mesh.boundaryMesh()[ipatch].faceCentres();
            volScalarField dtmp = getDistance2SingleWall(bfaceCenter, mesh);
	    
	    // assign to internal mesh
            forAll(distance, i)
            {
                distance[i] = Foam::min(distance[i], dtmp[i]);
            }
	    // assign to  boundary mesh
	    forAll(distance.boundaryField(), ipatch)
	    {
		forAll(distance.boundaryField()[ipatch], iface)
		{
		    distance.boundaryField()[ipatch][iface] 
		    = Foam::min( distance.boundaryField()[ipatch][iface],
				     dtmp.boundaryField()[ipatch][iface] );
		}
	    }
        }

    }
    return distance;
}

//RectangularMatrix <doubleScalar> &
void add_b (RectangularMatrix <doubleScalar> &net, const RectangularMatrix <doubleScalar> &b)
{
    for(int i = 0; i < net.n(); ++i)
    {
        for(int j = 0; j < net.m(); ++j)
        {
            net[i][j] += b[j][0];
        }
    }	
	//return net;
}

//RectangularMatrix <doubleScalar> &
void batch_norm (RectangularMatrix <doubleScalar> &net, 
			const RectangularMatrix <doubleScalar> &running_mean,
			const RectangularMatrix <doubleScalar> &running_var,
			const RectangularMatrix <doubleScalar> &gamma,
			const RectangularMatrix <doubleScalar> &beta,
			const scalar& eps_bn)
{
    for(int i = 0; i < net.n(); ++i)
    {
        for(int j = 0; j < net.m(); ++j)
        {
            net[i][j] -= running_mean[j][0];
            net[i][j] /= Foam::sqrt(running_var[j][0] + eps_bn);
            net[i][j] *= gamma[j][0];
            net[i][j] += beta [j][0];
        }
    }	
	//return net;
}

//RectangularMatrix <doubleScalar> &
void relu(RectangularMatrix <doubleScalar> &net)
{
	for(int i = 0; i < net.n(); ++i)
    {
        for(int j = 0; j < net.m(); ++j)
        {
            if(net[i][j] < 0)
                net[i][j] = 0;
        }
    }
	//return net;
}