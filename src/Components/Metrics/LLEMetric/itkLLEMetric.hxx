/*======================================================================

  This file is part of the elastix software.

  Copyright (c) Erasmus MC, Rotterdam. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef LLEMetric_HXX
#define LLEMetric_HXX
#include "itkLLEMetric.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "vnl/algo/vnl_cholesky.h"
#include "vnl/algo/vnl_symmetric_eigensystem.h"
#include "vnl/vnl_vector.txx"

namespace itk
{
/**
 * ******************* Constructor *******************
 */

template <class TFixedImage, class TMovingImage>
LLEMetric<TFixedImage,TMovingImage>
::LLEMetric()
{
    this->SetSubtractMean( true ),
    this->SetTransformIsStackTransform( true ),
    this->SetUseImageSampler( true );
    
    this->m_BinaryKNNTree  = 0;
    this->m_BinaryKNNTreeSearcher  = 0;

} // end constructor

/**
 * ************************ SetANNkDTree *************************
 */
    
template< class TFixedImage, class TMovingImage >
void
LLEMetric< TFixedImage, TMovingImage >
::SetANNkDTree( )
{
    typename ANNkDTreeType::Pointer tmpPtr = ANNkDTreeType::New();
        
    this->m_BinaryKNNTree  = tmpPtr;
        
} // end SetANNkDTree()

/**
 * ************************ SetANNBruteForceTree *************************
 */
    
template< class TFixedImage, class TMovingImage >
void
LLEMetric< TFixedImage, TMovingImage >
::SetANNBruteForceTree( )
{
    typename ANNkDTreeType::Pointer tmpPtr = ANNkDTreeType::New();
    
    this->m_BinaryKNNTree  = tmpPtr;
        
} // end SetANNBruteForceTree()

/**
 * ************************ SetANNStandardTreeSearch *************************
 */
    
template< class TFixedImage, class TMovingImage >
void
LLEMetric< TFixedImage, TMovingImage >
::SetANNStandardTreeSearch( unsigned int kNearestNeighbors)
    {
        typename ANNStandardTreeSearchType::Pointer tmpPtr = ANNStandardTreeSearchType::New();
        
        tmpPtr->SetKNearestNeighbors( kNearestNeighbors );
        
        tmpPtr->SetErrorBound( 0 );
        
        this->m_BinaryKNNTreeSearcher  = tmpPtr;
        
    } // end SetANNStandardTreeSearch()

/**
 * ******************* Initialize *******************
 */

template <class TFixedImage, class TMovingImage>
void
LLEMetric<TFixedImage,TMovingImage>
::Initialize(void) throw ( ExceptionObject )
{

    /** Initialize transform, interpolator, etc. */
    Superclass::Initialize();
    
    /** Retrieve slowest varying dimension and its size. */
    const unsigned int lastDim = this->GetFixedImage()->GetImageDimension() - 1;
    const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );

    if(!this->m_SampleLastDimensionRandomly)
    {
        this->m_NumSamplesLastDimension = lastDimSize;
    }

    this->m_RandomList.resize(this->m_NumSamplesLastDimension);
    
    if(!this->m_SampleLastDimensionRandomly)
    {
        for( unsigned int d = 0; d < lastDimSize; d++ )
        {
            this->m_RandomList[d]=d;
        }
    }
    else
    {
        this->SampleRandom( this->m_NumSamplesLastDimension, lastDimSize, this->m_RandomList );
    }

} // end Initialize


/**
 * ******************* PrintSelf *******************
 */

template < class TFixedImage, class TMovingImage>
void
LLEMetric<TFixedImage,TMovingImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
    Superclass::PrintSelf( os, indent );

} // end PrintSelf

/**
 * *************** EvaluateTransformJacobianInnerProduct ****************
 */

template < class TFixedImage, class TMovingImage >
void
LLEMetric<TFixedImage,TMovingImage>
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
 * ************************ ComputeListSamplesLastDim*************************
 */
    
template< class TFixedImage, class TMovingImage >
void
LLEMetric< TFixedImage, TMovingImage >
::ComputeListSamplesLastDim( const ListSamplePointer & listSamplesLastDim ) const
{
    const unsigned int lastDim = this->GetFixedImage()->GetImageDimension() - 1;
    const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );

    this->GetImageSampler()->Update();
    ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
    const unsigned long         nrOfRequestedSamples = sampleContainer->Size();

    typename ImageSampleContainerType::ConstIterator fiter;
    typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
    typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();
    
    this->m_NumberOfPixelsCounted = 0;

    listSamplesLastDim->SetMeasurementVectorSize(this->m_NumSamplesLastDimension);
    listSamplesLastDim->Resize(nrOfRequestedSamples);
    for( fiter = fbegin; fiter != fend; ++fiter )
    {
        FixedImagePointType fixedPoint = ( *fiter ).Value().m_ImageCoordinates;
        
        FixedImageContinuousIndexType voxelCoord;
        this->GetFixedImage()->TransformPhysicalPointToContinuousIndex( fixedPoint, voxelCoord );
        
        bool sampleOk =true;
        for( unsigned int s = 0; s < this->m_NumSamplesLastDimension; s++ )
        {
            RealType             movingImageValueTemp;
            MovingImagePointType mappedPoint;
            
            voxelCoord[ lastDim ] = this->m_RandomList[ s ];

            this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord, fixedPoint );
            
            sampleOk &= this->TransformPoint( fixedPoint, mappedPoint );
            
            if( sampleOk )
            {
                sampleOk &= this->IsInsideMovingMask( mappedPoint );
            }
            
            if( sampleOk )
            {
                sampleOk &= this->EvaluateMovingImageValueAndDerivative(mappedPoint, movingImageValueTemp, 0 );
            }
            
            if( sampleOk )
            {
                listSamplesLastDim->SetMeasurement(  this->m_NumberOfPixelsCounted, s, movingImageValueTemp );
            }
            
        }
        
        if(sampleOk)
        {
            this->m_NumberOfPixelsCounted++;
        }
        
    }
    
    listSamplesLastDim->SetActualSize( this->m_NumberOfPixelsCounted );
    
}

/**
 * ******************* GetValue *******************
 */

template <class TFixedImage, class TMovingImage>
typename LLEMetric<TFixedImage,TMovingImage>::MeasureType
LLEMetric<TFixedImage,TMovingImage>
::GetValue( const TransformParametersType & parameters ) const
{
    itkDebugMacro( "GetValue( " << parameters << " ) " );

    std::ofstream output;
    output.open("/Users/mp/Work/Data/Test_LLE/RegResults.txt",std::ios::app);

    const unsigned int lastDim = this->GetFixedImage()->GetImageDimension() - 1;
    const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );
    MeasureType measure = NumericTraits< MeasureType >::Zero;
    
    this->SetTransformParameters( parameters );

    ListSamplePointer listSamplesLastDim  = ListSampleType::New();

    this->ComputeListSamplesLastDim(listSamplesLastDim);
    this->m_BinaryKNNTree->SetSample( listSamplesLastDim );
    this->m_BinaryKNNTree->GenerateTree();
    
    this->m_BinaryKNNTreeSearcher->SetBinaryTree( this->m_BinaryKNNTree );
    
    MeasurementVectorType x,y,z;
    IndexArrayType        indices;
    DistanceArrayType     distances;

    vnl_matrix<double> allWeights(this->m_NumberOfPixelsCounted,this->m_NumberOfPixelsCounted,0);
    vnl_matrix<double> identity(this->m_NumberOfPixelsCounted,this->m_NumberOfPixelsCounted,0);
    vnl_matrix<double> sparseMatrix(this->m_NumberOfPixelsCounted,this->m_NumberOfPixelsCounted,0);
    vnl_vector<double> errors(this->m_NumberOfPixelsCounted,0);

    for( unsigned long i = 0; i < this->m_NumberOfPixelsCounted; i++ )
    {
        /** Get the i-th query point. */
        listSamplesLastDim->GetMeasurementVector(  i, z );
        
        /** Search for the K nearest neighbours of the current query point. */
        this->m_BinaryKNNTreeSearcher->Search(  z, indices, distances );
        
        vnl_vector<double> weights(this->m_NearestNeighbours, 1.0);
        vnl_vector<double> dots(this->m_NearestNeighbours);
        vnl_matrix<double> gramMatrix(this->m_NearestNeighbours,this->m_NearestNeighbours);

        double selfdot = dot_product(static_cast<vnl_vector<double> >(z),static_cast<vnl_vector<double> >(z ));
        for( unsigned long k = 0; k < this->m_NearestNeighbours; k++ )
        {
            listSamplesLastDim->GetMeasurementVector(  indices[k], x );
            dots[k] = dot_product(static_cast<vnl_vector<double> >(z),static_cast<vnl_vector<double> >(x));
        }

        for( unsigned long k = 0; k < this->m_NearestNeighbours; k++ )
        {
            listSamplesLastDim->GetMeasurementVector(  indices[k], x );
            for( unsigned long l = 0; l < this->m_NearestNeighbours; l++ )
            {
                listSamplesLastDim->GetMeasurementVector(  indices[l], y );
                gramMatrix(k,l) = selfdot - dots[k] - dots[l] + dot_product(static_cast<vnl_vector<double> >(x),static_cast<vnl_vector<double> >(y));
            }
        }
        
        gramMatrix.set_diagonal(gramMatrix.get_diagonal()+0.000001);
        vnl_cholesky eig( gramMatrix );
        eig.solve(weights,&weights);
        weights /= weights.sum();

        for( unsigned long k = 0; k < this->m_NearestNeighbours; k++ )
        {
            allWeights(indices[k],i) = weights[k];
        }
        
        identity(i,i) = 1;

    }

    sparseMatrix = (identity - allWeights)*(identity - allWeights).transpose();

    vnl_symmetric_eigensystem<double> eig(sparseMatrix);
    std::cout.precision(17);

    for( unsigned long k = 0; k < this->m_NumberOfEigenValues; k++ )
    {
        output << eig.get_eigenvalue(k) << " ";
    }
    
    for( unsigned long k = 0; k < this->m_NumberOfEigenValues; k++ )
    {
        errors += element_product((eig.get_eigenvector(k)-(eig.get_eigenvector(k)*allWeights)),(eig.get_eigenvector(k)-(eig.get_eigenvector(k)*allWeights)));
    }

    float accumulated = 0;
    for( unsigned long i = 0; i < this->m_NumberOfPixelsCounted; i++ )
    {
        accumulated += std::sqrt(errors(i));
    }
    output << accumulated/static_cast<float>(this->m_NumberOfPixelsCounted) << std::endl;
    return this->m_NumberOfPixelsCounted;
    
} // end GetValue

/**
 * ******************* GetDerivative *******************
 */

template < class TFixedImage, class TMovingImage>
void
LLEMetric<TFixedImage,TMovingImage>
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
LLEMetric<TFixedImage,TMovingImage>
::GetValueAndDerivative( const TransformParametersType & parameters,
                         MeasureType& value, DerivativeType& derivative ) const
{
    itkDebugMacro("GetValueAndDerivative( " << parameters << " ) ");
    /** Define derivative and Jacobian types. */
    typedef typename DerivativeType::ValueType        DerivativeValueType;
    typedef typename TransformJacobianType::ValueType TransformJacobianValueType;

} // end GetValueAndDerivative()

/**
 * ******************* SampleRandom *******************
 */
    
template< class TFixedImage, class TMovingImage >
void
LLEMetric< TFixedImage, TMovingImage >
::SampleRandom( const int n, const int m, std::vector< int > & numbers ) const
{
    numbers.clear();
        
    Statistics::MersenneTwisterRandomVariateGenerator::Pointer randomGenerator
        = Statistics::MersenneTwisterRandomVariateGenerator::GetInstance();
        
    for( int i = 0; i < n; ++i )
    {
        int randomNum = 0;
        do
        {
            randomNum = static_cast< int >( randomGenerator->GetVariateWithClosedRange( m ) );
        }
        while( find( numbers.begin(), numbers.end(), randomNum ) != numbers.end() );
        numbers.push_back( randomNum );
    }
} // end SampleRandom()

} // end namespace itk

#endif // ITKLLEMetric_HXX
