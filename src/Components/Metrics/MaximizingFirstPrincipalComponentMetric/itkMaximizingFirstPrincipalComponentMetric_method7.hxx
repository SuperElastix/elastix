/*======================================================================

  This file is part of the elastix software.

  Copyright (c) Erasmus MC, Rotterdam. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkmaximizingfirstprincipalcomponentmetric_method7_hxx
#define __itkmaximizingfirstprincipalcomponentmetric_method7_hxx
#include "itkMaximizingFirstPrincipalComponentMetric.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "vnl/algo/vnl_matrix_update.h"
#include "itkImage.h"
#include "vnl/algo/vnl_svd.h"
#include "vnl/vnl_trace.h"
#include "vnl/algo/vnl_symmetric_eigensystem.h"
#include <numeric>

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
        m_NumEigenValues( 1 ),
        m_UseDerivativeOfMean( false )

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
//    bool UseGetValueAndDerivative = false;

//    if(UseGetValueAndDerivative)
//    {
//        typedef typename DerivativeType::ValueType        DerivativeValueType;
//        const unsigned int P = this->GetNumberOfParameters();
//        MeasureType dummymeasure = NumericTraits< MeasureType >::Zero;
//        DerivativeType dummyderivative = DerivativeType( P );
//        dummyderivative.Fill( NumericTraits< DerivativeValueType >::Zero );

//        this->GetValueAndDerivative( parameters, dummymeasure, dummyderivative );
//        return dummymeasure;
//    }

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
    unsigned int N = this->m_NumberOfPixelsCounted;
    const unsigned int G = realNumLastDimPositions;

    MatrixType A( datablock.extract( N, G ) );

    /** Calculate mean of from columns */
    vnl_vector< RealType > mean( A.cols() );
    mean.fill( NumericTraits< RealType >::Zero );
    for( int i = 0; i < A.rows(); i++ )
    {
        for( int j = 0; j < A.cols(); j++)
        {
            mean(j) += A(i,j);
        }
    }
    mean /= RealType(A.rows());

    /** Calculate standard deviation from columns */
    vnl_vector< RealType > std( A.cols() );
    std.fill( NumericTraits< RealType >::Zero );
    for( int i = 0; i < A.rows(); i++ )
    {
        for( int j = 0; j < A.cols(); j++)
        {
            std(j) += pow((A(i,j)-mean(j)),2);
        }
    }
    std /= RealType(A.rows() - 1.0);

    for( int j = 0; j < A.cols(); j++)
    {
        std(j) = sqrt(std(j));
    }

    /** Subtract mean from columns */
    MatrixType AZscore( A.rows(), A.cols() );
    AZscore.fill( NumericTraits< RealType >::Zero );

    for (int i = 0; i < A.rows(); i++ )
    {
        for(int j = 0; j < A.cols(); j++)
        {
            AZscore(i,j) = (A(i,j)-mean(j))/std(j);
        }
    }

    /** Transpose of the matrix with mean subtracted */
    MatrixType AtZscore( AZscore.transpose() );

    /** Compute covariance matrix K */
    MatrixType K( (AtZscore*AZscore) );

    K /=  static_cast< RealType > ( A.rows() - static_cast< RealType > (1.0) );

    /** Compute first eigenvalue and eigenvector of the covariance matrix K */
    vnl_symmetric_eigensystem< RealType > eig( K );
    vnl_vector< RealType > eigenValues;
    eigenValues.set_size(K.cols());

    eigenValues.fill( itk::NumericTraits< RealType >::Zero );

    for(unsigned int i = 0; i < K.cols(); i++)
    {
        eigenValues(i) = eig.get_eigenvalue( i );
    }

    /** Compute sum of all eigenvalues = trace( K ) */
    RealType trace = itk::NumericTraits< RealType >::Zero;;
    for( int i = 0; i < K.rows(); i++ )
    {
        trace += K(i,i);
    }

    RealType sumEigenValuesUsed = itk::NumericTraits< RealType >::Zero;
    for(unsigned int i = 1; i < this->m_NumEigenValues+1; i++)
    {
        sumEigenValuesUsed += eig.get_eigenvalue(K.cols() - i);
    }

    measure = trace - sumEigenValuesUsed;

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
    MeasureType& value, DerivativeType& derivative ) const
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
    const unsigned int G = realNumLastDimPositions;

    /** The rows of the ImageSampleMatrix contain the samples of the images of the stack */
    unsigned int NumberOfSamples = sampleContainer->Size();
    MatrixType datablock( NumberOfSamples, G );

    /** Initialize dummy loop variables */
    unsigned int pixelIndex = 0;

    /** Initialize image sample matrix . */
    datablock.fill( itk::NumericTraits< RealType >::Zero );

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

    for ( fiter = fbegin; fiter != fend; ++fiter )
    {
        /** Read fixed coordinates. */
        FixedImagePointType fixedPoint = (*fiter).Value().m_ImageCoordinates;

        /** Transform sampled point to voxel coordinates. */
        FixedImageContinuousIndexType voxelCoord;
        this->GetFixedImage()->TransformPhysicalPointToContinuousIndex( fixedPoint, voxelCoord );

        const unsigned int G = lastDimPositions.size();
        unsigned int numSamplesOk = 0;

        /** Loop over t */
        for ( unsigned int d = 0; d < G; ++d )
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

        if( numSamplesOk == G )
        {
            SamplesOK.push_back(fixedPoint);
            pixelIndex++;
            this->m_NumberOfPixelsCounted++;
        }

    }/** end first loop over image sample container */

    /** Check if enough samples were valid. */
    this->CheckNumberOfSamples(	sampleContainer->Size(), this->m_NumberOfPixelsCounted );
    this->m_NumberOfSamples = this->m_NumberOfPixelsCounted;
    unsigned int N = pixelIndex;

    MatrixType A( datablock.extract( N, G ) );

    /** Calculate mean of from columns */
    vnl_vector< RealType > mean( A.cols() );
    mean.fill( NumericTraits< RealType >::Zero );
    for( int i = 0; i < A.rows(); i++ )
    {
        for( int j = 0; j < A.cols(); j++)
        {
            mean(j) += A(i,j);
        }
    }
    mean /= RealType(A.rows());

    /** Calculate standard deviation from columns */
    vnl_vector< RealType > std( A.cols() );
    std.fill( NumericTraits< RealType >::Zero );
    for( int i = 0; i < A.rows(); i++ )
    {
        for( int j = 0; j < A.cols(); j++)
        {
            std(j) += pow((A(i,j)-mean(j)),2);
        }
    }
    std /= RealType( A.rows() - 1.0 );

    for( int j = 0; j < A.cols(); j++)
    {
        std(j) = sqrt(std(j));
    }

    /** Subtract mean from columns */
    MatrixType AZscore( A.rows(), A.cols() );
    AZscore.fill( NumericTraits< RealType >::Zero );

    for (int i = 0; i < A.rows(); i++ )
    {
        for(int j = 0; j < A.cols(); j++)
        {
            AZscore(i,j) = (A(i,j)-mean(j))/std(j);
        }
    }

    MatrixType Amm( A.rows(), A.cols() );
    Amm.fill( NumericTraits< RealType >::Zero );
    for( unsigned int i = 0; i < A.rows(); i++)
    {
        for( unsigned int j = 0; j < A.cols(); j++)
        {
            Amm(i,j) = A(i,j) - mean(j);
        }
    }

    MatrixType Atmm( Amm.transpose() );

    /** Transpose of the matrix with mean subtracted */
    MatrixType AtZscore( AZscore.transpose() );

    /** Compute covariance matrix K */
    MatrixType K( AtZscore*AZscore );

    K /= static_cast< RealType > ( A.rows()  - 1.0 );

    /** Compute first eigenvalue and eigenvector of the covariance matrix K */
    vnl_symmetric_eigensystem< RealType > eig( K );

    /** Compute sum of all eigenvalues = trace( K ) */
    RealType trace = itk::NumericTraits< RealType >::Zero;
    for( int i = 0; i < K.rows(); i++ )
    {
        trace += K(i,i);
    }

    RealType sumEigenValuesUsed = itk::NumericTraits< DerivativeValueType >::Zero;
    for(unsigned int i = 1; i < this->m_NumEigenValues+1; i++)
    {
        sumEigenValuesUsed += eig.get_eigenvalue(K.cols() - i);
    }

    vnl_vector< RealType > eigenValues( K.rows() );
    eigenValues.fill( 0.0 );
    for(unsigned int i = 0; i < K.rows(); i++)
    {
        eigenValues[ i ] = eig.get_eigenvalue( i );
    }
    this->m_eigenValues = eigenValues;

    MatrixType eigenVectorMatrix( A.cols(), this->m_NumEigenValues );
    for(unsigned int i = 1; i < this->m_NumEigenValues+1; i++)
    {
        eigenVectorMatrix.set_column(i-1, (eig.get_eigenvector(K.cols() - i)).normalize() );
    }

    MatrixType eigenVectorMatrixTranspose( eigenVectorMatrix.transpose() );

    this->m_firstEigenVector = eigenVectorMatrix.get_column( 0 );
    this->m_secondEigenVector = eigenVectorMatrix.get_column( 1 );
    this->m_thirdEigenVector = eigenVectorMatrix.get_column( 2 );
    this->m_fourthEigenVector = eigenVectorMatrix.get_column( 3 );
    this->m_fifthEigenVector = eigenVectorMatrix.get_column( 4 );
    this->m_sixthEigenVector = eigenVectorMatrix.get_column( 5 );
    this->m_seventhEigenVector = eigenVectorMatrix.get_column( 6 );

    /** Create variables to store intermediate results in. */
    TransformJacobianType jacobian;
    DerivativeType dMTdmu;
    DerivativeType imageJacobian( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
    std::vector<NonZeroJacobianIndicesType> nzjis( G, NonZeroJacobianIndicesType() );

    /** Sub components of metric derivative */
    vnl_vector< DerivativeValueType > tracevKvdmu( P );
    vnl_vector< DerivativeValueType > vSinvdSinvdmuAtmmAmmv( P );
    vnl_vector< DerivativeValueType > dSigmainvdmu_part1( G );
    DerivativeMatrixType vdAdmu( this->m_NumEigenValues, G*P );

    /** initialize */
    vdAdmu.fill( itk::NumericTraits< DerivativeValueType >::Zero );
    tracevKvdmu.fill( itk::NumericTraits< DerivativeValueType >::Zero );
    dSigmainvdmu_part1.fill( itk::NumericTraits< DerivativeValueType >::Zero );
    vSinvdSinvdmuAtmmAmmv.fill( itk::NumericTraits< DerivativeValueType >::Zero );

    /** Components for derivative of mean */
    vnl_vector< DerivativeValueType > meanvSinvdSinvdmuAtmmAmmv( P );
    DerivativeMatrixType meandAdmu( G, P );
    DerivativeMatrixType vmeandAdmu( this->m_NumEigenValues, G*P );
    vmeandAdmu.fill( itk::NumericTraits< DerivativeValueType >::Zero );
    meandAdmu.fill( itk::NumericTraits< DerivativeValueType >::Zero );
    meanvSinvdSinvdmuAtmmAmmv.fill( itk::NumericTraits< DerivativeValueType >::Zero );

    MatrixType K2eigenVectorMatrix( eigenVectorMatrixTranspose*AtZscore*Amm );

    unsigned int startSamplesOK;
    startSamplesOK = 0;

    for(unsigned int d = 0; d < G; d++)
    {
        dSigmainvdmu_part1[ d ] = pow(std[ d ],-3);
    }

    dSigmainvdmu_part1 /= -RealType(A.rows()-1.0);

    MatrixType eigenVectorMatrixTransposeAtZscore( eigenVectorMatrixTranspose*AtZscore );

    /** Second loop over fixed image samples. */
    for ( pixelIndex = 0; pixelIndex < SamplesOK.size(); ++pixelIndex )
    {
        /** Read fixed coordinates. */
        FixedImagePointType fixedPoint = SamplesOK[ startSamplesOK ];
        startSamplesOK++;

        /** Transform sampled point to voxel coordinates. */
        FixedImageContinuousIndexType voxelCoord;
        this->GetFixedImage()->TransformPhysicalPointToContinuousIndex( fixedPoint, voxelCoord );

        const unsigned int G = lastDimPositions.size();

        for ( unsigned int d = 0; d < G; ++d )
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

            /** Get the TransformJacobian dT/dmu */
            this->EvaluateTransformJacobian( fixedPoint, jacobian, nzjis[ d ] );

            /** Compute the innerproduct (dM/dx)^T (dT/dmu). */
            this->EvaluateTransformJacobianInnerProduct(
                        jacobian, movingImageDerivative, imageJacobian );

            /** Store values. */
            dMTdmu = imageJacobian;

            /** build metric derivative components */
            if(this->m_UseDerivativeOfMean)
            {
                for( unsigned int p = 0; p < nzjis[ d ].size(); ++p)
                {
                    meandAdmu[ d ][ nzjis[ d ][ p ] ] += ( dMTdmu[ p ]/std[ d ] )/A.rows();
                }
            }

            for(unsigned int z = 0; z < this->m_NumEigenValues; z++)
            {
                for( unsigned int p = 0; p < nzjis[ d ].size(); ++p)
                {
                    vdAdmu[ z ][ d + nzjis[ d ][ p ]*G ] +=
                            eigenVectorMatrixTransposeAtZscore[ z ][ pixelIndex ]*(dMTdmu[ p ]/std[ d ]);
                    vSinvdSinvdmuAtmmAmmv[ nzjis[ d ][ p ] ] += K2eigenVectorMatrix[ z ][ d ]
                            *dSigmainvdmu_part1[ d ]*Atmm[ d ][ pixelIndex ]*dMTdmu[ p ]*eigenVectorMatrix[ d ][ z ];
                }
            }
        }//end loop over t
    } // end second for loop over sample container

    if(this->m_UseDerivativeOfMean)
    {
        for(unsigned int i = 0; i < A.rows(); i++)
        {
            for (unsigned int d = 0; d < G; ++d )
            {
                for(unsigned int z = 0; z < this->m_NumEigenValues; z++)
                {
                    for(unsigned int p = 0; p < P; ++p )
                    {
                        vmeandAdmu[ z ][ d + G*p ] += eigenVectorMatrixTransposeAtZscore[ z ][ i ]
                                *meandAdmu[ d ][ p ];
                        meanvSinvdSinvdmuAtmmAmmv[ p ] += K2eigenVectorMatrix[ z ][ d ]
                                *dSigmainvdmu_part1[ d ]*Atmm[ d ][ i ]*meandAdmu[ d ][ p ]*eigenVectorMatrix[ d ][ z ];
                    }
                }
            }
        }
    }

    for(unsigned int p = 0; p < P; p++)
    {
        tracevKvdmu[ p ] = vnl_trace< DerivativeValueType > ((vdAdmu-vmeandAdmu).extract(this->m_NumEigenValues,
                                                                                        G,0,p*G)*eigenVectorMatrix);
    }

    tracevKvdmu += vSinvdSinvdmuAtmmAmmv;

    tracevKvdmu *= static_cast < DerivativeValueType > (2.0)
        / ( static_cast < DerivativeValueType > (A.rows()) -
        static_cast < DerivativeValueType >(1.0) ); //normalize

    measure = trace - sumEigenValuesUsed;
    derivative = -tracevKvdmu;

    /** Subtract mean from derivative elements. */
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
          mean /= static_cast< RealType >( lastDimGridSize );

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
        mean /= static_cast< RealType >( lastDimSize );

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
    this->m_normdCdmu.set_size(lastDimSize);
    this->m_normdCdmu.fill(0.0);
    unsigned int ind = 0;
    for ( unsigned int t = 0; t < lastDimSize; ++t )
    {
        const unsigned int startc = (this->GetNumberOfParameters() / lastDimSize)*t;
        for ( unsigned int c = startc; c < startc + (this->GetNumberOfParameters() / lastDimSize); ++c )
        {
         this->m_normdCdmu[ ind ] += pow(derivative[ c ],2);
        }
        ++ind;
    }


    for(unsigned int index = 0; index < this->m_normdCdmu.size(); index++)
    {
        this->m_normdCdmu[index] = sqrt(this->m_normdCdmu.get(index));
    }

    this->m_normdCdmu /= static_cast< RealType >( this->GetNumberOfParameters() / lastDimSize );


    /** Return the measure value. */
    value = measure;

    } // end GetValueAndDerivative()

} // end namespace itk

#endif // end #ifndef _itkmaximizingfirstprincipalcomponentmetric_method7_hxx


    
