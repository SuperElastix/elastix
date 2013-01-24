/*======================================================================

This file is part of the elastix software.

Copyright (c) Erasmus MC, Rotterdam. All rights reserved.
See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef ITKSUMOFPAIRWISENORMALIZEDCORRELATIONCOEFFICIENTSMETRIC_HXX
#define ITKSUMOFPAIRWISENORMALIZEDCORRELATIONCOEFFICIENTSMETRIC_HXX
#include "itkMaximizingFirstPrincipalComponentMetric.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "vnl/algo/vnl_matrix_update.h"
#include "itkImage.h"
#include "vnl/algo/vnl_svd.h"
#include "vnl/vnl_trace.h"
#include "vnl/algo/vnl_symmetric_eigensystem.h"
#include <numeric>
#include <fstream>
#include <iostream>
#include <iomanip>


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
    m_TransformIsStackTransform( false )
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
            std(j) += pow((A(i,j)-mean(j)),2);
        }
    }
    std /= double(A.rows() - 1.0);

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

    measure = 1.0-(K.fro_norm()/double(realNumLastDimPositions));

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
            std(j) += pow((A(i,j)-mean(j)),2);
        }
    }
    std /= double( A.rows() - 1.0 );

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

    /** Create variables to store intermediate results in. */
    TransformJacobianType jacobian;
    DerivativeType imageJacobian( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
    std::vector<NonZeroJacobianIndicesType> nzjis( realNumLastDimPositions, NonZeroJacobianIndicesType() );

    /** Sub components of metric derivative */
    DerivativeMatrixType meandAdmu( realNumLastDimPositions, P ); // mean of a column of the derivative of A
    DerivativeMatrixType dSigmainvdmu( realNumLastDimPositions, P );
    DerivativeMatrixType meandSigmainvdmu( realNumLastDimPositions, P );
    DerivativeMatrixType dSigmainvtdmu( realNumLastDimPositions, P );
    DerivativeMatrixType meandSigmainvtdmu( realNumLastDimPositions, P );

    vnl_vector< DerivativeValueType > sumAtZscoredAzscoredmu( P );
    vnl_vector< DerivativeValueType > sumAtZscoreAmmdSigmadmu( P );
    vnl_vector< DerivativeValueType > sumdAtZscoredmuAzscore( P );
    vnl_vector< DerivativeValueType > sumdSigmadmuAtmmAzscore( P );
    vnl_vector< DerivativeValueType > meansumAtZscoredAzscoredmu( P );
    vnl_vector< DerivativeValueType > meansumdAtZscoredmuAzscore( P );
    vnl_vector< DerivativeValueType > dSigmainvdmu_part1( realNumLastDimPositions );
    
    DerivativeType dMTdmu;

    /** initialize */
    meandAdmu.fill( 0.0 );
    dSigmainvdmu.fill( 0.0 );
    dSigmainvtdmu.fill( 0.0 );
    sumdSigmadmuAtmmAzscore.fill( 0.0 );
    sumdAtZscoredmuAzscore.fill( 0.0 );
    sumAtZscoredAzscoredmu.fill( 0.0 );
    sumAtZscoreAmmdSigmadmu.fill( 0.0 );
    meandSigmainvdmu.fill( 0.0 );
    meansumAtZscoredAzscoredmu.fill( 0.0 );
    meansumdAtZscoredmuAzscore.fill( 0.0 );
    meandSigmainvtdmu.fill( 0.0 );
    dSigmainvdmu_part1.fill( 0.0 );

    unsigned int startSamplesOK;
    startSamplesOK = 0;

		ofstream file;
		file.open("K.txt");
		file << K << std::endl;
		file.close();
    for(unsigned int d = 0; d < realNumLastDimPositions; d++)
    {
        dSigmainvdmu_part1[ d ] = pow(std[ d ],-3);
    }

    dSigmainvdmu_part1 /= -double(A.rows()-1.0);

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

            /** Store values. */
            dMTdmu = imageJacobian;

            /** build metric derivative components */
            for( unsigned int p = 0; p < nzjis[ d ].size(); ++p)
            {
                dSigmainvdmu[ d ][ nzjis[ d ][ p ] ] += dSigmainvdmu_part1[ d ]*Atmm[ d ][ pixelIndex ]*dMTdmu[ p ];
                dSigmainvtdmu[ d ][ nzjis[ d ][ p ] ] += dSigmainvdmu_part1[ d ]*dMTdmu[ p ]*Amm[ pixelIndex ][ d ];
                for(unsigned int l = 0; l < realNumLastDimPositions; l++)
                {
                    sumAtZscoredAzscoredmu[ nzjis[ d ][ p ] ] += K[ d ][ l ]*AtZscore[ l ][ pixelIndex ]*(dMTdmu[ p ]/std[ d ]);
                    sumdAtZscoredmuAzscore[ nzjis[ d ][ p ] ] += K[ d ][ l ]*(dMTdmu[ p ]/std[ d ])*AZscore[ pixelIndex ][ l ];
                }

                meandAdmu[ d ][ nzjis[ d ][ p ] ] += (dMTdmu[ p ]/std[ d ])/A.rows();
            }

        } // end loop over t
    } // end second for loop over sample container

    for(unsigned int i = 0; i < A.rows(); i++)
    {
        for (unsigned int d = 0; d < realNumLastDimPositions; ++d )
        {
            for(unsigned int p = 0; p < P; ++p )
            {
                meandSigmainvdmu[ d ][ p ] += dSigmainvdmu_part1[ d ]*Atmm[ d ][ i ]*meandAdmu[ d ][ p ];
                meandSigmainvtdmu[ d ][ p ] += dSigmainvdmu_part1[ d ]*meandAdmu[ d ][ p ]*Amm[ i ][ d ];
                for(unsigned int l  = 0; l < realNumLastDimPositions; l++)
                {
                    meansumAtZscoredAzscoredmu[ p ] += K[ d ][ l ]*AtZscore[ d ][ i ]*meandAdmu[ l ][ p ];
                    meansumdAtZscoredmuAzscore[ p ] += K[ d ][ l ]*meandAdmu[ l ][ p ]*AZscore[ i ][ d ];
                }
            }
        }
    }

    dSigmainvdmu -= meandSigmainvdmu;
    dSigmainvtdmu -= meandSigmainvtdmu;
    sumdAtZscoredmuAzscore -= meansumdAtZscoredmuAzscore;
    sumAtZscoredAzscoredmu -= meansumAtZscoredAzscoredmu;

    MatrixType AtZscoreAmm( AtZscore*Amm );
    MatrixType AtmmAZscore( Atmm*AZscore );

    for(unsigned int l = 0; l < realNumLastDimPositions; l++)
    {
        for(unsigned int d = 0; d < realNumLastDimPositions; d++)
        {
            for(unsigned int p = 0; p < P; p++)
            {
                sumAtZscoreAmmdSigmadmu[ p ] += K[ d ][ l ]*AtZscoreAmm[ l ][ d ]*dSigmainvdmu[ d ][ p ];
                sumdSigmadmuAtmmAzscore[ p ] += K[ d ][ l ]*dSigmainvtdmu[ l ][ p ]*AtmmAZscore[ l ][ d ];
            }
        }
    }

    vnl_vector< DerivativeValueType > sumKdabsKdmu( P );

    sumKdabsKdmu = sumdSigmadmuAtmmAzscore + sumdAtZscoredmuAzscore +
            sumAtZscoredAzscoredmu + sumAtZscoreAmmdSigmadmu;

    sumKdabsKdmu *= static_cast < DerivativeValueType > (1.0)
            / ( static_cast < DerivativeValueType > (A.rows()) -
                static_cast < DerivativeValueType >(1.0) ); //normalize

    derivative = -sumKdabsKdmu/(K.fro_norm()*double(realNumLastDimPositions));

    measure = 1.0-(K.fro_norm()/double(realNumLastDimPositions));

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

#endif // ITKSUMOFPAIRWISENORMALIZEDCORRELATIONCOEFFICIENTSMETRIC_HXX
