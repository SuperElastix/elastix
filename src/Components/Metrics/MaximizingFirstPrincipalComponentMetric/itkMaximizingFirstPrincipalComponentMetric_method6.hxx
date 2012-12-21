/*======================================================================

  This file is part of the elastix software.

  Copyright (c) Erasmus MC, Rotterdam. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkMaximizingFirstPrincipalComponentMetric_method6_hxx
#define __itkMaximizingFirstPrincipalComponentMetric_method6_hxx

#include "itkMaximizingFirstPrincipalComponentMetric.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "vnl/algo/vnl_matrix_update.h"
#include "itkImage.h"
#include "vnl/algo/vnl_svd.h"
#include "vnl/algo/vnl_symmetric_eigensystem.h"
#include <numeric>
#include <fstream>
#include <iostream>

using namespace std;

namespace itk
{
/**
 * ******************* Constructor *******************
 */

  template <class TFixedImage, class TMovingImage>
    MaximizingFirstPrincipalComponentMetric<TFixedImage,TMovingImage>
      ::MaximizingFirstPrincipalComponentMetric():
        m_SampleLastDimensionRandomly( false ),
        m_NumSamplesLastDimension( 10 ),
        m_SubtractMean( false ),
        m_TransformIsStackTransform( false ),
        m_Alpha( 1.0 )

  {
    this->SetUseImageSampler( true );
    this->SetUseFixedImageLimiter( false );
    this->SetUseMovingImageLimiter( false );
  } // end constructor

 /**
 * ******************* Initialize *******************
 */

  template <class TFixedImage, class TMovingImage>
    void
    MaximizingFirstPrincipalComponentMetric<TFixedImage,TMovingImage>
    ::Initialize(void) throw ( ExceptionObject )
  {

        /** Initialize transform, interpolator, etc. */
    Superclass::Initialize();

    /** Retrieve slowest varying dimension and its size. */
    const unsigned int lastDim = this->GetFixedImage()->GetImageDimension() - 1;
    const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );

    /** Check num last samples. */
    if ( this->m_NumSamplesLastDimension > lastDimSize )
    {
      this->m_NumSamplesLastDimension = lastDimSize;
    }

  } // end Initialize


/**
 * ******************* PrintSelf *******************
 */

  template < class TFixedImage, class TMovingImage>
    void
    MaximizingFirstPrincipalComponentMetric<TFixedImage,TMovingImage>
    ::PrintSelf(std::ostream& os, Indent indent) const
  {
    Superclass::PrintSelf( os, indent );

  } // end PrintSelf


/**
 * ******************* SampleRandom *******************
 */

  template < class TFixedImage, class TMovingImage>
  void
     MaximizingFirstPrincipalComponentMetric<TFixedImage,TMovingImage>
    ::SampleRandom ( const int n, const int m, std::vector<int> & numbers ) const
  {
    /** Empty list of last dimension positions. */
    numbers.clear();

    /** Initialize random number generator. */
    Statistics::MersenneTwisterRandomVariateGenerator::Pointer randomGenerator = Statistics::MersenneTwisterRandomVariateGenerator::GetInstance();

    /** Sample additional at fixed timepoint. */
    for ( unsigned int i = 0; i < m_NumAdditionalSamplesFixed; ++i )
    {
      numbers.push_back( this->m_ReducedDimensionIndex );
    }

    /** Get n random samples. */
    for ( int i = 0; i < n; ++i )
    {
      int randomNum = 0;
      do
      {
        randomNum = static_cast<int>( randomGenerator->GetVariateWithClosedRange( m ) );
      } while ( find( numbers.begin(), numbers.end(), randomNum ) != numbers.end() );
      numbers.push_back( randomNum );
    }
  } // end SampleRandom

/**
 * *************** EvaluateTransformJacobianInnerProduct ****************
 */

  template < class TFixedImage, class TMovingImage >
    void
    MaximizingFirstPrincipalComponentMetric<TFixedImage,TMovingImage>
    ::EvaluateTransformJacobianInnerProduct(
    const TransformJacobianType & jacobian,
    const MovingImageDerivativeType & movingImageDerivative,
    DerivativeType & imageJacobian ) const
  {
    typedef typename TransformJacobianType::const_iterator JacobianIteratorType;
    typedef typename DerivativeType::iterator              DerivativeIteratorType;
    JacobianIteratorType jac = jacobian.begin();
    imageJacobian.Fill( 0.0 );
    const unsigned int sizeImageJacobian = imageJacobian.GetSize();
    for ( unsigned int dim = 0; dim < FixedImageDimension; dim++ )
    {
      const double imDeriv = movingImageDerivative[ dim ];
      DerivativeIteratorType imjac = imageJacobian.begin();

      for ( unsigned int mu = 0; mu < sizeImageJacobian; mu++ )
      {
        (*imjac) += (*jac) * imDeriv;
        ++imjac;
        ++jac;
      }
    }
  } // end EvaluateTransformJacobianInnerProduct

 /**
 * ******************* GetValue *******************
 */

  template <class TFixedImage, class TMovingImage>
    typename MaximizingFirstPrincipalComponentMetric<TFixedImage,TMovingImage>::MeasureType
    MaximizingFirstPrincipalComponentMetric<TFixedImage,TMovingImage>
    ::GetValue( const TransformParametersType & parameters ) const
  {
    itkDebugMacro( "GetValue( " << parameters << " ) " );

    /** Make sure the transform parameters are up to date. */
    this->SetTransformParameters( parameters );

        /** Initialize some variables */
        this->m_NumberOfPixelsCounted = 0;
        MeasureType measure = NumericTraits< MeasureType >::Zero;

        /** Update the imageSampler and get a handle to the sample container. */
        this->GetImageSampler()->Update();
        ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

        /** Create iterator over the sample container. */
        typename ImageSampleContainerType::ConstIterator fiter;
        typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
        typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

        /** Retrieve slowest varying dimension and its size. */
        const unsigned int lastDim = this->GetFixedImage()->GetImageDimension() - 1;
        const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );
        const unsigned int numLastDimSamples = this->m_NumSamplesLastDimension;

        typedef vnl_matrix< RealType > MatrixType;

        /** Get real last dim samples. */
        const unsigned int realNumLastDimPositions = this->m_SampleLastDimensionRandomly ? this->m_NumSamplesLastDimension + this->m_NumAdditionalSamplesFixed : lastDimSize;

        /** The rows of the ImageSampleMatrix contain the samples of the images of the stack */
        unsigned int NumberOfSamples = sampleContainer->Size();
        MatrixType datablock( NumberOfSamples, realNumLastDimPositions );

        /** Vector containing last dimension positions to use: initialize on all positions when random sampling turned off. */
        std::vector<int> lastDimPositions;

        /** Determine random last dimension positions if needed. */

        if ( this->m_SampleLastDimensionRandomly )
        {
            SampleRandom( this->m_NumSamplesLastDimension, lastDimSize, lastDimPositions );
        }
        else
        {
            for ( unsigned int i = 0; i < lastDimSize; ++i )
            {
                lastDimPositions.push_back( i );
            }
        }

        /** Initialize dummy loop variable */
        unsigned int pixelIndex = 0;

        std::vector < FixedImagePointType > SamplesOK;

        /** Initialize image sample matrix . */
        datablock.fill( itk::NumericTraits< RealType>::Zero );

        for ( fiter = fbegin; fiter != fend; ++fiter )
        {
            /** Read fixed coordinates. */
            FixedImagePointType fixedPoint = (*fiter).Value().m_ImageCoordinates;

            /** Transform sampled point to voxel coordinates. */
            FixedImageContinuousIndexType voxelCoord;
            this->GetFixedImage()->TransformPhysicalPointToContinuousIndex( fixedPoint, voxelCoord );

            unsigned int numSamplesOk = 0;

            /** Loop over t */
            for ( unsigned int d = 0; d < realNumLastDimPositions; ++d )
            {
                /** Initialize some variables. */
                RealType movingImageValue;
                MovingImagePointType mappedPoint;

                /** Set fixed point's last dimension to lastDimPosition. */
                voxelCoord[ lastDim ] = lastDimPositions[ d ];

                /** Transform sampled point back to world coordinates. */
                this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord, fixedPoint );

                /** Transform point and check if it is inside the B-spline support region. */
                bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

                /** Check if point is inside mask. */
                if ( sampleOk )
                {
                    sampleOk = this->IsInsideMovingMask( mappedPoint );
                }

                if ( sampleOk )
                {
                    sampleOk = this->EvaluateMovingImageValueAndDerivative(
                        mappedPoint, movingImageValue, 0 );
                }

                if( sampleOk )
                {
                    numSamplesOk++;
                    datablock( pixelIndex, d ) = movingImageValue;
                }

            } /** end loop over t */

            if( numSamplesOk == realNumLastDimPositions )
            {
                pixelIndex++;
                this->m_NumberOfPixelsCounted++;
            }

        }/** end first loop over image sample container */

        /** Check if enough samples were valid. */
        this->CheckNumberOfSamples(
            NumberOfSamples, this->m_NumberOfPixelsCounted );

        MatrixType A( datablock.extract( pixelIndex, realNumLastDimPositions ) );


        /** Calculate mean of from columns */
        vnl_vector< double > mean( A.cols() );
        mean.fill( NumericTraits< double >::Zero );
        for( int i = 0; i < A.rows(); i++ )
        {
            for( int j = 0; j < A.cols(); j++)
            {
                mean(j) += A(i,j);
            }
        }
        mean /= double(A.rows());

        /** Calculate standard deviation from columns */
        vnl_vector< double > std( A.cols() );
        std.fill( NumericTraits< double >::Zero );
        for( int i = 0; i < A.rows(); i++ )
        {
            for( int j = 0; j < A.cols(); j++)
            {
                std(j) += pow((A(i,j)-mean(j)),2)/double((A.rows()-1.0));
            }
        }

        for( int j = 0; j < A.cols(); j++)
        {
            std(j) = sqrt(std(j));
        }


        /** Subtract mean from columns */
        MatrixType Azscore( A.rows(), A.cols() );
        Azscore.fill( NumericTraits< RealType >::Zero );
        for (int i = 0; i < A.rows(); i++ )
        {
            for(int j = 0; j < A.cols(); j++)
            {
                Azscore(i,j) = (A(i,j)-mean(j))/std(j);
            }
        }

        /** Transpose of the matrix with mean subtracted */
        MatrixType Atzscore( Azscore.transpose() );

        /** Compute covariance matrix K */
        MatrixType K( (Atzscore*Azscore) );

        K /= ( static_cast< RealType > (A.rows()) - static_cast< RealType > (1.0) );

         /** Compute first eigenvalue and eigenvector of the covariance matrix K */
        vnl_symmetric_eigensystem< RealType > eig( K );

        RealType e1 = eig.get_eigenvalue( K.cols() - 1 ); // Highest eigenvalue of K
        RealType e2 = eig.get_eigenvalue( K.cols() - 2 ); // Highest eigenvalue of K
        RealType e3 = eig.get_eigenvalue( K.cols() - 3 ); // Highest eigenvalue of K
        RealType e4 = eig.get_eigenvalue( K.cols() - 4 ); // Highest eigenvalue of K
        RealType e5 = eig.get_eigenvalue( K.cols() - 5 ); // Highest eigenvalue of K
        RealType e6 = eig.get_eigenvalue( K.cols() - 6 ); // Highest eigenvalue of K
        RealType e7 = eig.get_eigenvalue( K.cols() - 7 ); // Highest eigenvalue of K

        /** Compute sum of all eigenvalues = trace( K ) */
        RealType trace = 0.0;
        for( int i = 0; i < K.rows(); i++ )
        {
            trace += K(i,i);
        }

				vnl_vector<RealType> v1(eig.get_eigenvector(K.cols()-1));

        //measure = 1000*abs(v1(0))+trace - (this->m_Alpha*e1+e2+e3+e4+e5+e6);
        measure = trace - (this->m_Alpha*e1+e2+e3+e4+e5+e6);
        //measure = 1.0 - (e1+e2+e3+e4+e5+e6+e7)/trace;

        vnl_vector<double> eigenValues;
        eigenValues.set_size(K.cols());

        eigenValues.fill(0.0);

        for(unsigned int i = 0; i < K.cols(); i++)
        {
           eigenValues(i) = eig.get_eigenvalue( i );
        }
        this->m_firstEigenVector = eig.get_eigenvector( K.cols() - 1) ;
        this->m_secondEigenVector = eig.get_eigenvector( K.cols() - 2) ;
        this->m_thirdEigenVector = eig.get_eigenvector( K.cols() - 3) ;
        this->m_fourthEigenVector = eig.get_eigenvector( K.cols() - 4) ;
        this->m_fifthEigenVector = eig.get_eigenvector( K.cols() - 5) ;
        this->m_sixthEigenVector = eig.get_eigenvector( K.cols() - 6) ;
        this->m_seventhEigenVector = eig.get_eigenvector( K.cols() - 7) ;

        this->m_eigenValues = eigenValues;

        /** Return the measure value. */
        return measure;

  } // end GetValue

/**
 * ******************* GetDerivative *******************
 */

  template < class TFixedImage, class TMovingImage>
    void
    MaximizingFirstPrincipalComponentMetric<TFixedImage,TMovingImage>
    ::GetDerivative( const TransformParametersType & parameters,
    DerivativeType & derivative ) const
  {
    /** When the derivative is calculated, all information for calculating
     * the metric value is available. It does not cost anything to calculate
     * the metric value now. Therefore, we have chosen to only implement the
     * GetValueAndDerivative(), supplying it with a dummy value variable. */
    MeasureType dummyvalue = NumericTraits< MeasureType >::Zero;
    this->GetValueAndDerivative(parameters, dummyvalue, derivative);

  } // end GetDerivative

/**
 * ******************* GetValueAndDerivative *******************
 */

    template <class TFixedImage, class TMovingImage>
    void
   MaximizingFirstPrincipalComponentMetric<TFixedImage,TMovingImage>
    ::GetValueAndDerivative( const TransformParametersType & parameters,
    MeasureType& value, DerivativeType& derivative) const
  {
    itkDebugMacro("GetValueAndDerivative( " << parameters << " ) ");

    /** Define derivative and Jacobian types. */
    typedef typename DerivativeType::ValueType        DerivativeValueType;
    typedef typename TransformJacobianType::ValueType TransformJacobianValueType;

    /** Initialize some variables */
        const unsigned int P = this->GetNumberOfParameters();
        this->m_NumberOfPixelsCounted = 0;
        MeasureType measure = NumericTraits< MeasureType >::Zero;
        derivative = DerivativeType( P );
        derivative.Fill( NumericTraits< DerivativeValueType >::Zero );

        /** Make sure the transform parameters are up to date. */
        this->SetTransformParameters( parameters );

        /** Update the imageSampler and get a handle to the sample container. */
        this->GetImageSampler()->Update();
        ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

        /** Create iterator over the sample container. */
        typename ImageSampleContainerType::ConstIterator fiter;
        typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
        typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

        /** Retrieve slowest varying dimension and its size. */
        const unsigned int lastDim = this->GetFixedImage()->GetImageDimension() - 1;
        const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );
        const unsigned int numLastDimSamples = this->m_NumSamplesLastDimension;

        typedef vnl_matrix< RealType >                  MatrixType;
        typedef vnl_matrix< DerivativeValueType > DerivativeMatrixType;

        std::vector< FixedImagePointType > SamplesOK;

        /** Get real last dim samples. */
        const unsigned int realNumLastDimPositions = this->m_SampleLastDimensionRandomly ? this->m_NumSamplesLastDimension + this->m_NumAdditionalSamplesFixed : lastDimSize;

        /** The rows of the ImageSampleMatrix contain the samples of the images of the stack */
        unsigned int NumberOfSamples = sampleContainer->Size();
        MatrixType datablock( NumberOfSamples, realNumLastDimPositions );

        /** Initialize dummy loop variables */
        unsigned int pixelIndex = 0;

        /** Initialize image sample matrix . */
        datablock.fill( itk::NumericTraits< double >::Zero );

        /** Determine random last dimension positions if needed. */
        /** Vector containing last dimension positions to use: initialize on all positions when random sampling turned off. */
        std::vector<int> lastDimPositions;
        if ( this->m_SampleLastDimensionRandomly )
        {
            SampleRandom( this->m_NumSamplesLastDimension, lastDimSize, lastDimPositions );
        }
        else
        {
            for ( unsigned int i = 0; i < lastDimSize; ++i )
            {
                lastDimPositions.push_back( i );
            }
        }

        //elxout<< "start loop over sample container 1" << std::endl;
        for ( fiter = fbegin; fiter != fend; ++fiter )
        {
            /** Read fixed coordinates. */
            FixedImagePointType fixedPoint = (*fiter).Value().m_ImageCoordinates;

            /** Transform sampled point to voxel coordinates. */
            FixedImageContinuousIndexType voxelCoord;
            this->GetFixedImage()->TransformPhysicalPointToContinuousIndex( fixedPoint, voxelCoord );

            const unsigned int realNumLastDimPositions = lastDimPositions.size();
            unsigned int numSamplesOk = 0;

            /** Loop over t */
            for ( unsigned int d = 0; d < realNumLastDimPositions; ++d )
            {
                /** Initialize some variables. */
                RealType movingImageValue;
                MovingImagePointType mappedPoint;
                MovingImageDerivativeType movingImageDerivative;

                /** Set fixed point's last dimension to lastDimPosition. */
                voxelCoord[ lastDim ] = lastDimPositions[ d ];

                /** Transform sampled point back to world coordinates. */
                this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord, fixedPoint );

                /** Transform point and check if it is inside the B-spline support region. */
                bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

                /** Check if point is inside mask. */
                if( sampleOk )
                {
                    sampleOk = this->IsInsideMovingMask( mappedPoint );
                }

                if( sampleOk )
                {
                    sampleOk = this->EvaluateMovingImageValueAndDerivative(
                    mappedPoint, movingImageValue, 0 );
                }

                if( sampleOk )
                {
                    numSamplesOk++;
                    datablock( pixelIndex, d ) = movingImageValue;

                }// end if sampleOk

            } // end loop over t

            if( numSamplesOk == realNumLastDimPositions )
            {
                SamplesOK.push_back(fixedPoint);
                pixelIndex++;
                this->m_NumberOfPixelsCounted++;
            }

        }/** end first loop over image sample container */

        /** Check if enough samples were valid. */
        this->CheckNumberOfSamples(	sampleContainer->Size(), this->m_NumberOfPixelsCounted );

        MatrixType A( datablock.extract( pixelIndex, realNumLastDimPositions ) );

        /** Calculate mean of from columns */
        vnl_vector< double > mean( A.cols() );
        mean.fill( NumericTraits< double >::Zero );
        for( int i = 0; i < A.rows(); i++ )
        {
            for( int j = 0; j < A.cols(); j++)
            {
                mean(j) += A(i,j);
            }
        }
        mean /= double(A.rows());

        /** Calculate standard deviation from columns */
        vnl_vector< double > std( A.cols() );
        std.fill( NumericTraits< double >::Zero );
        for( int i = 0; i < A.rows(); i++ )
        {
            for( int j = 0; j < A.cols(); j++)
            {
                std(j) += pow((A(i,j)-mean(j)),2)/double((A.rows()-1.0));
            }
        }

        for( int j = 0; j < A.cols(); j++)
        {
            std(j) = sqrt(std(j));
        }

        /** Subtract mean from columns */
        MatrixType Azscore( A.rows(), A.cols() );
        Azscore.fill( NumericTraits< RealType >::Zero );
        for (int i = 0; i < A.rows(); i++ )
        {
            for(int j = 0; j < A.cols(); j++)
            {
                Azscore(i,j) = (A(i,j)-mean(j))/std(j);
            }
        }

        /** Transpose of the matrix with mean subtracted */
        MatrixType Atzscore( Azscore.transpose() );

        /** Compute covariance matrix K */
        MatrixType K( (Atzscore*Azscore) );

        K /= ( static_cast< RealType > (A.rows()) - static_cast< RealType > (1.0) );

        /** Compute first eigenvalue and eigenvector of the covariance matrix K */
        vnl_symmetric_eigensystem< RealType > eig( K );
        RealType e1 = eig.get_eigenvalue( K.cols() - 1 ); // Highest eigenvalue of K
        RealType e2 = eig.get_eigenvalue( K.cols() - 2 ); // Highest eigenvalue of K
        RealType e3 = eig.get_eigenvalue( K.cols() - 3 ); // Highest eigenvalue of K
        RealType e4 = eig.get_eigenvalue( K.cols() - 4 ); // Highest eigenvalue of K
        RealType e5 = eig.get_eigenvalue( K.cols() - 5 ); // Highest eigenvalue of K
        RealType e6 = eig.get_eigenvalue( K.cols() - 6 ); // Highest eigenvalue of K
        RealType e7 = eig.get_eigenvalue( K.cols() - 7 ); // Highest eigenvalue of K

        vnl_vector< RealType > FirstEigenvector = eig.get_eigenvector(K.cols()-1);
        vnl_vector< RealType > v1 = FirstEigenvector.normalize(); // Highest eigenvector of A'*A
        vnl_vector< RealType > SecondEigenvector = eig.get_eigenvector(K.cols()-2);
        vnl_vector< RealType > v2 = SecondEigenvector.normalize(); // Highest eigenvector of A'*A
        vnl_vector< RealType > ThirdEigenvector = eig.get_eigenvector(K.cols()-3);
        vnl_vector< RealType > v3 = ThirdEigenvector.normalize(); // Highest eigenvector of A'*A
        vnl_vector< RealType > FourthEigenvector = eig.get_eigenvector(K.cols()-4);
        vnl_vector< RealType > v4 = FourthEigenvector.normalize(); // Highest eigenvector of A'*A
        vnl_vector< RealType > FifthEigenvector = eig.get_eigenvector(K.cols()-5);
        vnl_vector< RealType > v5 = FifthEigenvector.normalize(); // Highest eigenvector of A'*A
        vnl_vector< RealType > SixthEigenvector = eig.get_eigenvector(K.cols()-6);
        vnl_vector< RealType > v6 = SixthEigenvector.normalize(); // Highest eigenvector of A'*A
        vnl_vector< RealType > SeventhEigenvector = eig.get_eigenvector(K.cols()-7);
        vnl_vector< RealType > v7 = SeventhEigenvector.normalize(); // Highest eigenvector of A'*A

        /** Compute sum of all eigenvalues = trace( K ) */
        double trace = 0.0;
        for( int i = 0; i < K.rows(); i++ )
        {
            trace += K(i,i);
        }

        vnl_vector<double> eigenValues;
        eigenValues.set_size(K.cols());

        eigenValues.fill(0.0);

        for(unsigned int i = 0; i < K.cols(); i++)
        {
           eigenValues(i) = eig.get_eigenvalue( i );
        }
        this->m_firstEigenVector = eig.get_eigenvector( K.cols() - 1 );
        this->m_secondEigenVector = eig.get_eigenvector( K.cols() - 2) ;
        this->m_thirdEigenVector = eig.get_eigenvector( K.cols() - 3) ;
        this->m_fourthEigenVector = eig.get_eigenvector( K.cols() - 4) ;
        this->m_fifthEigenVector = eig.get_eigenvector( K.cols() - 5) ;
        this->m_sixthEigenVector = eig.get_eigenvector( K.cols() - 6) ;
        this->m_seventhEigenVector = eig.get_eigenvector( K.cols() - 7) ;
        this->m_eigenValues = eigenValues;

        /** Create variables to store intermediate results in. */
    TransformJacobianType jacobian;
    DerivativeType imageJacobian( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
        std::vector<NonZeroJacobianIndicesType> nzjis ( realNumLastDimPositions, NonZeroJacobianIndicesType() );

        /** Sub components of metric derivative */
    vnl_vector< DerivativeValueType > dKiidmu( P ); //Trace of derivative of covariance matrix
    vnl_vector< DerivativeValueType > AtdAdmuii( P ); //Trace of AtMinusMean * dAdmu
    vnl_vector< DerivativeValueType > v1Kv1dmu( P ); //v1 * derivative covariance matrix * v1
    vnl_vector< DerivativeValueType > v2Kv2dmu( P ); //v1 * derivative covariance matrix * v2
    vnl_vector< DerivativeValueType > v3Kv3dmu( P ); //v1 * derivative covariance matrix * v3
    vnl_vector< DerivativeValueType > v4Kv4dmu( P ); //v1 * derivative covariance matrix * v4
    vnl_vector< DerivativeValueType > v5Kv5dmu( P ); //v1 * derivative covariance matrix * v5
    vnl_vector< DerivativeValueType > v6Kv6dmu( P ); //v1 * derivative covariance matrix * v6
    vnl_vector< DerivativeValueType > v7Kv7dmu( P ); //v1 * derivative covariance matrix * v7

    DerivativeMatrixType dAdmu_v1( pixelIndex, P ); //dAdmu * v1
    DerivativeMatrixType dAdmu_v2( pixelIndex, P ); //dAdmu * v2
    DerivativeMatrixType dAdmu_v3( pixelIndex, P ); //dAdmu * v3
    DerivativeMatrixType dAdmu_v4( pixelIndex, P ); //dAdmu * v4
    DerivativeMatrixType dAdmu_v5( pixelIndex, P ); //dAdmu * v5
    DerivativeMatrixType dAdmu_v6( pixelIndex, P ); //dAdmu * v6
    DerivativeMatrixType dAdmu_v7( pixelIndex, P ); //dAdmu * v7

    //DerivativeMatrixType dAdmu_v( A.cols()*pixelIndex, P );

    /** initialize */
    dKiidmu.fill ( itk::NumericTraits< DerivativeValueType >::Zero );
    AtdAdmuii.fill ( itk::NumericTraits< DerivativeValueType >::Zero );

    dAdmu_v1.fill ( itk::NumericTraits< DerivativeValueType >::Zero );
    dAdmu_v2.fill ( itk::NumericTraits< DerivativeValueType >::Zero );
    dAdmu_v3.fill ( itk::NumericTraits< DerivativeValueType >::Zero );
    dAdmu_v4.fill ( itk::NumericTraits< DerivativeValueType >::Zero );
    dAdmu_v5.fill ( itk::NumericTraits< DerivativeValueType >::Zero );
    dAdmu_v6.fill ( itk::NumericTraits< DerivativeValueType >::Zero );
    dAdmu_v7.fill ( itk::NumericTraits< DerivativeValueType >::Zero );

    //dAdmu_v.fill ( itk::NumericTraits< DerivativeValueType >::Zero );

    v1Kv1dmu.fill ( itk::NumericTraits< DerivativeValueType >::Zero );
    v2Kv2dmu.fill ( itk::NumericTraits< DerivativeValueType >::Zero );
    v3Kv3dmu.fill ( itk::NumericTraits< DerivativeValueType >::Zero );
    v4Kv4dmu.fill ( itk::NumericTraits< DerivativeValueType >::Zero );
    v5Kv5dmu.fill ( itk::NumericTraits< DerivativeValueType >::Zero );
    v6Kv6dmu.fill ( itk::NumericTraits< DerivativeValueType >::Zero );
    v7Kv7dmu.fill ( itk::NumericTraits< DerivativeValueType >::Zero );


    DerivativeType dMTdmu;
    dMTdmu.fill( itk::NumericTraits<RealType>::Zero );

    //unsigned int NumSamplesUsed;
    unsigned int startSamplesOK;

    //NumSamplesUsed = SamplesOK.size()/realNumLastDimPositions;
    startSamplesOK = 0;

    /** Second loop over fixed image samples. */
    for ( pixelIndex = 0; pixelIndex < SamplesOK.size(); ++pixelIndex )
    {
        /** Read fixed coordinates. */
        FixedImagePointType fixedPoint = SamplesOK[ startSamplesOK ];
        startSamplesOK++;

        /** Transform sampled point to voxel coordinates. */
        FixedImageContinuousIndexType voxelCoord;
        this->GetFixedImage()->TransformPhysicalPointToContinuousIndex( fixedPoint, voxelCoord );

        const unsigned int realNumLastDimPositions = lastDimPositions.size();
        unsigned int numSamplesOk = 0;

        for ( unsigned int d = 0; d < realNumLastDimPositions; ++d )
        {
            /** Initialize some variables. */
            RealType movingImageValue;
            MovingImagePointType mappedPoint;
            MovingImageDerivativeType movingImageDerivative;

                /** Set fixed point's last dimension to lastDimPosition. */
                voxelCoord[ lastDim ] = lastDimPositions[ d ];

                /** Transform sampled point back to world coordinates. */
                this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord, fixedPoint );
                this->TransformPoint( fixedPoint, mappedPoint );
                this->EvaluateMovingImageValueAndDerivative(
                                            mappedPoint, movingImageValue, &movingImageDerivative );

                movingImageDerivative /= std(d);
                /** Get the TransformJacobian dT/dmu */
                this->EvaluateTransformJacobian( fixedPoint, jacobian, nzjis[ d ] );

                /** Compute the innerproduct (dM/dx)^T (dT/dmu). */
                this->EvaluateTransformJacobianInnerProduct(
                   jacobian, movingImageDerivative, imageJacobian );

            /** Store values. */
            dMTdmu = imageJacobian;

            /** build metric derivative components */
            for( unsigned int p = 0; p < nzjis[ d ].size(); ++p)
            {
                dAdmu_v1[ pixelIndex ][ nzjis[ d ][ p ] ] += dMTdmu[ p ]*v1[ d ];
                dAdmu_v2[ pixelIndex ][ nzjis[ d ][ p ] ] += dMTdmu[ p ]*v2[ d ];
                dAdmu_v3[ pixelIndex ][ nzjis[ d ][ p ] ] += dMTdmu[ p ]*v3[ d ];
                dAdmu_v4[ pixelIndex ][ nzjis[ d ][ p ] ] += dMTdmu[ p ]*v4[ d ];
                dAdmu_v5[ pixelIndex ][ nzjis[ d ][ p ] ] += dMTdmu[ p ]*v5[ d ];
                dAdmu_v6[ pixelIndex ][ nzjis[ d ][ p ] ] += dMTdmu[ p ]*v6[ d ];
                dAdmu_v7[ pixelIndex ][ nzjis[ d ][ p ] ] += dMTdmu[ p ]*v7[ d ];
                AtdAdmuii[ nzjis[ d ][ p ] ] += Atzscore[ d ][ pixelIndex ]*dMTdmu[ p ];
			
                //for(unsigned int s = 0; s < A.cols(); s++)
                //{
                //	dAdmu_v[ (s+1)*pixelIndex ][ nzjis[ d ][ p ] ] += dMTdmu[ p ]*eig.get_eigenvector( K.cols() - (s+1) );
                //}
								
            }
        } // end loop over t
    } // end second for loop over sample container

        v1Kv1dmu = v1*Atzscore*dAdmu_v1;
        v1Kv1dmu *= static_cast < DerivativeValueType > (2.0)
            / ( static_cast < DerivativeValueType > (A.rows()) -
            static_cast < DerivativeValueType > (1.0) ); //normalize
        v2Kv2dmu = v2*Atzscore*dAdmu_v2;
        v2Kv2dmu *= static_cast < DerivativeValueType > (2.0)
            / ( static_cast < DerivativeValueType > (A.rows()) -
            static_cast < DerivativeValueType > (1.0) ); //normalize
        v3Kv3dmu = v3*Atzscore*dAdmu_v3;
        v3Kv3dmu *= static_cast < DerivativeValueType > (2.0)
            / ( static_cast < DerivativeValueType > (A.rows()) -
            static_cast < DerivativeValueType > (1.0) ); //normalize
        v4Kv4dmu = v4*Atzscore*dAdmu_v4;
        v4Kv4dmu *= static_cast < DerivativeValueType > (2.0)
            / ( static_cast < DerivativeValueType > (A.rows()) -
            static_cast < DerivativeValueType > (1.0) ); //normalize
        v5Kv5dmu = v5*Atzscore*dAdmu_v5;
        v5Kv5dmu *= static_cast < DerivativeValueType > (2.0)
            / ( static_cast < DerivativeValueType > (A.rows()) -
            static_cast < DerivativeValueType > (1.0) ); //normalize
        v6Kv6dmu = v6*Atzscore*dAdmu_v6;
        v6Kv6dmu *= static_cast < DerivativeValueType > (2.0)
            / ( static_cast < DerivativeValueType > (A.rows()) -
            static_cast < DerivativeValueType > (1.0) ); //normalize
        v7Kv7dmu = v7*Atzscore*dAdmu_v7;
        v7Kv7dmu *= static_cast < DerivativeValueType > (2.0)
            / ( static_cast < DerivativeValueType > (A.rows()) -
            static_cast < DerivativeValueType > (1.0) ); //normalize

        dKiidmu = AtdAdmuii;
        dKiidmu *= static_cast < DerivativeValueType > (2.0)
            / ( static_cast < DerivativeValueType > (A.rows()) -
            static_cast < DerivativeValueType >(1.0) ); //normalize

//        DerivativeMatrixType C(A.rows(), P);
				
//        C.Fill(0.0);
//        for(unsigned int k = 1; k < C.rows(); k++ );
//        {
//            DerivativeMatrixType dAdmu_vi( dAdmu_v.extract(A.rows(),P,k*A.rows(),0) );
//            C += v1*Atzscore*dAdmu_vi/(eig.get_eigenvalue(K.cols() - (k+1))-e1);
//        }

//        DerivativeMatrixType C_part2(A.rows(), P);
//        unsigned int m = 0;
//        for(unsigned int k = 1; k < C.rows(); k++ );
//        {
//            C_part2 += v1(m)*eig.get_eigenvector( K.cols() - (k+1) )*Atzscore*dAdmu_v1/(e1 - eig.get_eigenvalue( K.cols()-(k+1)) );
//            m++;
//        }

//        C_part2 /= v1(0);
//        C -= C_part2;

//        vnl_vector<DerivativeValueType > dv1dmu(P);
//        for(unsigned int s = 0; s < C.rows(); s++)
//        {
//            for(unsigned int p = 0; p < P; p++)
//            {
//                dv1dmu[ p ] += C[ s ][ p ];
//            }
//        }

//        dv1dmu *= v1(0);

       	double regularisationFirstEigenValue = this->m_Alpha;

       //measure = abs(v1(0))*1000 + trace - (this->m_Alpha*e1+e2+e3+e4+e5+e6);

       measure = trace - (this->m_Alpha*e1+e2+e3+e4+e5+e6);
       derivative = dKiidmu -
                (this->m_Alpha*v1Kv1dmu+v2Kv2dmu+v3Kv3dmu+v4Kv4dmu+v5Kv5dmu+v6Kv6dmu);


        //** Subtract mean from derivative elements. */
    if ( this->m_SubtractMean )
    {
      if ( ! this->m_TransformIsStackTransform )
      {
        /** Update derivative per dimension.
         * Parameters are ordered xxxxxxx yyyyyyy zzzzzzz ttttttt and
         * per dimension xyz.
         */
        const unsigned int lastDimGridSize = this->m_GridSize[ lastDim ];
        const unsigned int numParametersPerDimension
          = this->GetNumberOfParameters() / this->GetMovingImage()->GetImageDimension();
        const unsigned int numControlPointsPerDimension = numParametersPerDimension / lastDimGridSize;
        DerivativeType mean ( numControlPointsPerDimension );
        for ( unsigned int d = 0; d < this->GetMovingImage()->GetImageDimension(); ++d )
        {
          /** Compute mean per dimension. */
          mean.Fill( 0.0 );
          const unsigned int starti = numParametersPerDimension * d;
          for ( unsigned int i = starti; i < starti + numParametersPerDimension; ++i )
          {
            const unsigned int index = i % numControlPointsPerDimension;
            mean[ index ] += derivative[ i ];
          }
          mean /= static_cast< double >( lastDimGridSize );

          /** Update derivative for every control point per dimension. */
          for ( unsigned int i = starti; i < starti + numParametersPerDimension; ++i )
          {
            const unsigned int index = i % numControlPointsPerDimension;
            derivative[ i ] -= mean[ index ];
          }
        }
      }
      else
      {
        /** Update derivative per dimension.
         * Parameters are ordered x0x0x0y0y0y0z0z0z0x1x1x1y1y1y1z1z1z1 with
         * the number the time point index.
         */
        const unsigned int numParametersPerLastDimension = this->GetNumberOfParameters() / lastDimSize;
        DerivativeType mean ( numParametersPerLastDimension );
        mean.Fill( 0.0 );

        /** Compute mean per control point. */
        for ( unsigned int t = 0; t < lastDimSize; ++t )
        {
          const unsigned int startc = numParametersPerLastDimension * t;
          for ( unsigned int c = startc; c < startc + numParametersPerLastDimension; ++c )
          {
            const unsigned int index = c % numParametersPerLastDimension;
            mean[ index ] += derivative[ c ];
          }
        }
        mean /= static_cast< double >( lastDimSize );

        /** Update derivative per control point. */
        for ( unsigned int t = 0; t < lastDimSize; ++t )
        {
          const unsigned int startc = numParametersPerLastDimension * t;
          for ( unsigned int c = startc; c < startc + numParametersPerLastDimension; ++c )
          {
            const unsigned int index = c % numParametersPerLastDimension;
            derivative[ c ] -= mean[ index ];
          }
        }
      }
    }

		/** Compute norm of transform parameters per image */
		unsigned int ind = 0;
  	for ( unsigned int t = 0; t < this->GetNumberOfParameters(); ++t )
  	{
    	const unsigned int startc = (this->GetNumberOfParameters() / lastDimSize)*t;
    	for ( unsigned int c = startc; c < startc + (this->GetNumberOfParameters() / lastDimSize); ++c )
    	{
         this->m_normdCdmu[ ind ] += pow(derivative[ c ],2);		
				 ++ind;	 
      }
    }


		for(unsigned int index = 0; index < this->m_normdCdmu.size(); index++)
		{
				this->m_normdCdmu[index] = sqrt(this->m_normdCdmu.get(index));
		}

  	this->m_normdCdmu /= static_cast< double >( this->GetNumberOfParameters() / lastDimSize );



    /** Return the measure value. */
    value = measure;

    } // end GetValueAndDerivative()

} // end namespace itk

#endif // end #ifndef _itkMaximizingFirstPrincipalComponentMetric_method6b_hxx

