#ifndef ITKMAXIMIZINGFIRSTPRINCIPALCOMPONENTMETRIC_METHOD5B_HXX
#define ITKMAXIMIZINGFIRSTPRINCIPALCOMPONENTMETRIC_METHOD5B_HXX
#include "itkMaximizingFirstPrincipalComponentMetric.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "vnl/algo/vnl_matrix_update.h"
#include "itkImage.h"
#include "vnl/algo/vnl_svd.h"
#include "vnl/algo/vnl_symmetric_eigensystem.h"
#include <numeric>
#include <fstream>
#include <iostream>

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
        m_Zscore( false ),
        m_Alpha(1.0)
                //m_NumEigenValues( 1 )
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
        MatrixType datablock( realNumLastDimPositions, NumberOfSamples );

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
                    datablock( d, pixelIndex ) = movingImageValue;
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

        MatrixType A( datablock.extract( realNumLastDimPositions, pixelIndex ) );

        /** Calculate mean of the rows */
        vnl_vector< RealType > meanrows( A.rows() );
        meanrows.fill( NumericTraits< double >::Zero );
        for( unsigned int i = 0; i < A.rows(); i++ )
        {
            for( unsigned int j = 0; j < A.cols(); j++)
            {
                meanrows(i) += A(i,j);
            }
        }
        meanrows /= double(A.cols());

        /** Calculate standard deviation of the rows */
        vnl_vector< double > std( A.rows() );
        std.fill( NumericTraits< double >::Zero );
        for( int i = 0; i < A.rows(); i++ )
        {
            for( int j = 0; j < A.cols(); j++)
            {
                std(i) += pow((A(i,j)-meanrows(i)),2)/double((A.cols()-1.0));
            }
        }

        for( int i = 0; i < A.rows(); i++)
        {
            std(i) = sqrt(std(i));
        }

        /** Z-score A */
        if(this->m_Zscore)
        {
            for (int i = 0; i < A.rows(); i++ )
            {
                for(int j = 0; j < A.cols(); j++)
                {
                    A(i,j) = (A(i,j)-meanrows(i))/std(i);
                }
            }
        }


        /** Calculate mean of from columns */
        vnl_vector< RealType > meancols( A.cols() );
        meancols.fill( NumericTraits< double >::Zero );
        for( unsigned int i = 0; i < A.rows(); i++ )
        {
            for( unsigned int j = 0; j < A.cols(); j++)
            {
                meancols(j) += A(i,j);
            }
        }
				meancols /= double(A.rows());

        /** Subtract mean from columns */
        MatrixType AMinusMean( A.rows(), A.cols() );
        AMinusMean.fill( NumericTraits< RealType >::Zero );
        for (unsigned int i = 0; i < A.rows(); i++ )
        {
            for (unsigned int j = 0; j < A.cols(); j++)
            {
                AMinusMean(i,j) = A(i,j)-meancols(j);
            }
        }

        /** Transpose of the matrix with mean subtracted */
        MatrixType AtMinusMean( AMinusMean.transpose() );

        /** Compute covariance matrix K */
        MatrixType K( (AMinusMean*AtMinusMean) );

        K /= ( static_cast< RealType > (A.rows()) - static_cast< RealType > (1.0) );

        /** Compute first eigenvalue and eigenvector of the covariance matrix K */
        vnl_symmetric_eigensystem< RealType > eig( K );
        RealType e1 = eig.get_eigenvalue( K.rows() - 1 ); // Highest eigenvalue of K

        /** Compute sum of all eigenvalues = trace( K ) */
        RealType trace = 0.0;
        for( int i = 0; i < K.rows(); i++ )
        {
            trace += K(i,i);
        }

       measure = trace - e1;

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

    //dummyimageMatrix.fill( NumericTraits< double >::Zero);
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
        MatrixType datablock( realNumLastDimPositions, NumberOfSamples );

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
                    datablock( d, pixelIndex ) = movingImageValue;

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

    MatrixType A( datablock.extract( realNumLastDimPositions, pixelIndex ) );

    /** Calculate mean of the rows */
    vnl_vector< RealType > meanrows( A.rows() );
    meanrows.fill( NumericTraits< double >::Zero );
    for( unsigned int i = 0; i < A.rows(); i++ )
    {
        for( unsigned int j = 0; j < A.cols(); j++)
        {
            meanrows(i) += A(i,j);
        }
    }
    meanrows /= double(A.cols());

    /** Calculate standard deviation of the rows */
    vnl_vector< double > std( A.rows() );
    std.fill( NumericTraits< double >::Zero );
    for( int i = 0; i < A.rows(); i++ )
    {
        for( int j = 0; j < A.cols(); j++)
        {
            std(i) += pow((A(i,j)-meanrows(i)),2)/double((A.cols()-1.0));
        }
    }

    for( int i = 0; i < A.rows(); i++)
    {
        std(i) = sqrt(std(i));
    }

    /** Z-score A */
    if(this->m_Zscore)
    {
        for (int i = 0; i < A.rows(); i++ )
        {
            for(int j = 0; j < A.cols(); j++)
            {
                A(i,j) = (A(i,j)-meanrows(i))/std(i);
            }
        }
    }


    /** Calculate mean of from columns */
    vnl_vector< double > meancols( A.cols() );
    meancols.fill( NumericTraits< double >::Zero );
    for( int i = 0; i < A.rows(); i++ )
    {
        for( int j = 0; j < A.cols(); j++)
        {
            meancols(j) += A(i,j);
        }
    }
		meancols /= double(A.rows());

    /** Subtract mean from columns */
    MatrixType AMinusMean( A.rows(), A.cols() );
    AMinusMean.fill( NumericTraits< RealType >::Zero );
    for (int i = 0; i < A.rows(); i++ )
    {
        for(int j = 0; j < A.cols(); j++)
        {
            AMinusMean(i,j) = A(i,j)-meancols(j);
        }
    }

    /** Transpose of the matrix with mean subtracted */
    MatrixType AtMinusMean( AMinusMean.transpose() );


    /** Compute covariance matrix K */
    MatrixType K( (AMinusMean*AtMinusMean) );

    K /= ( static_cast< RealType > (A.rows()) - static_cast< RealType > (1.0) );

    /** Compute first eigenvalue and eigenvector of the covariance matrix K */
    vnl_symmetric_eigensystem< RealType > eig( K );
    RealType e1 = eig.get_eigenvalue( K.cols() - 1 ); // Highest eigenvalue of K
    vnl_vector< RealType > FirstEigenvector = eig.get_eigenvector(K.cols()-1);
    vnl_vector< RealType > v1 = (AtMinusMean*FirstEigenvector).normalize(); // Highest eigenvector of A'*A

    /** Compute sum of all eigenvalues = trace( K ) */
    double trace = 0.0;
    for( int i = 0; i < K.rows(); i++ )
    {
        trace += K(i,i);
    }

    /** Create variables to store intermediate results in. */
    TransformJacobianType jacobian;
    DerivativeType imageJacobian( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
    std::vector<NonZeroJacobianIndicesType> nzjis ( realNumLastDimPositions, NonZeroJacobianIndicesType() );

    /** Sub components of metric derivative */
    vnl_vector< DerivativeValueType > meandAdmu( P ); // mean of a column of the derivative of A
    vnl_vector< DerivativeValueType > dKiidmu( P ); //Trace of derivative of covariance matrix
    vnl_vector< DerivativeValueType > AtdAdmuii( P ); //Trace of AtMinusMean * dAdmu
    vnl_vector< DerivativeValueType > v1Kv1dmu( P ); //v1 * derivative covariance matrix * v1
    vnl_vector< DerivativeValueType > meanAtdAdmuii( P ); //mean of trace of AtMinusMean * dAdmu

    DerivativeMatrixType dAdmu_v1( realNumLastDimPositions, P ); //dAdmu * v1
    DerivativeMatrixType meandAdmu_v1( realNumLastDimPositions, P ); //meandAdmu * v

    /** initialize */
    dKiidmu.fill ( itk::NumericTraits< DerivativeValueType >::Zero );
    AtdAdmuii.fill ( itk::NumericTraits< DerivativeValueType >::Zero );
    dAdmu_v1.fill ( itk::NumericTraits< DerivativeValueType >::Zero );
    meandAdmu_v1.fill ( itk::NumericTraits< DerivativeValueType >::Zero );
    v1Kv1dmu.fill ( itk::NumericTraits< DerivativeValueType >::Zero );
    meandAdmu.fill( itk::NumericTraits< DerivativeValueType >::Zero );
    meanAtdAdmuii.fill( itk::NumericTraits< DerivativeValueType >::Zero );

    DerivativeType dMTdmu;
    dMTdmu.fill( itk::NumericTraits<RealType>::Zero );

    unsigned int startSamplesOK;

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

            /** Get the TransformJacobian dT/dmu */
            this->EvaluateTransformJacobian( fixedPoint, jacobian, nzjis[ d ] );

            /** Compute the innerproduct (dM/dx)^T (dT/dmu). */
            this->EvaluateTransformJacobianInnerProduct(
                        jacobian, movingImageDerivative, imageJacobian );

            if(this->m_Zscore)
            {
                movingImageDerivative/=std(d);
            }

            /** Store values. */
            dMTdmu = imageJacobian;

            /** build metric derivative components */
            for( unsigned int p = 0; p < nzjis[ d ].size(); ++p)
            {
                dAdmu_v1[ d ][ nzjis[ d ][ p ] ] += dMTdmu[ p ]*v1[ pixelIndex ];
                meandAdmu[ nzjis[ d ][ p ] ] += dMTdmu[ p ]/realNumLastDimPositions;
                AtdAdmuii[ nzjis[ d ][ p ] ] += AtMinusMean[ pixelIndex ][ d ]*dMTdmu[ p ];
            }


        } // end loop over t

        for (unsigned int d = 0; d < realNumLastDimPositions; ++d )
        {
            for(unsigned int p = 0; p < P; ++p )
            {
                meandAdmu_v1[ d ][ p ] += meandAdmu[ p ]*v1[ pixelIndex ];
                meanAtdAdmuii[ p ] += AtMinusMean[ pixelIndex ][ d ]*meandAdmu[ p ];
            }
        }


    } // end second for loop over sample container

    v1Kv1dmu = v1*AtMinusMean*( dAdmu_v1 - meandAdmu_v1 );
    v1Kv1dmu *= static_cast < DerivativeValueType > (2.0)
            / ( static_cast < DerivativeValueType > (A.rows()) -
                static_cast < DerivativeValueType > (1.0) ); //normalize

    dKiidmu = AtdAdmuii - meanAtdAdmuii;
    dKiidmu *= static_cast < DerivativeValueType > (2.0)
            / ( static_cast < DerivativeValueType > (A.rows()) -
                static_cast < DerivativeValueType >(1.0) ); //normalize

    measure = trace - e1;
    derivative = dKiidmu - v1Kv1dmu;

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


    /** Return the measure value. */
    value = measure;

    } // end GetValueAndDerivative()

} // end namespace itk

#endif // ITKMAXIMIZINGFIRSTPRINCIPALCOMPONENTMETRIC_METHOD5B_HXX
