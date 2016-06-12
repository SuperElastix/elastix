/*======================================================================

  This file is part of the elastix software.

  Copyright (c) Erasmus MC, Rotterdam. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef LeigsMetric_HXX
#define LeigsMetric_HXX
#include "itkLeigsMetric.h"
#include <cmath>
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "vnl/algo/vnl_cholesky.h"
#include "vnl/algo/vnl_generalized_eigensystem.h"
#include "vnl/algo/vnl_sparse_symmetric_eigensystem.h"
#include "vnl/vnl_vector.txx"
#include "vnl/vnl_sparse_matrix.txx"

namespace itk
{
/**
 * ******************* Constructor *******************
 */

template <class TFixedImage, class TMovingImage>
LeigsMetric<TFixedImage,TMovingImage>
::LeigsMetric()
{
    this->SetSubtractMean( true ),
    this->SetTransformIsStackTransform( true ),
    this->SetUseImageSampler( true );
    
    this->m_BinaryKNNTree  = 0;
    this->m_BinaryKNNTreeSearcher  = 0;

    this->m_LeigsMetricThreaderParameters.m_Metric = this;

    this->m_LeigsMetricComputeDerivativePerThreadVariables = NULL;
    this->m_LeigsMetricComputeDerivativePerThreadVariablesSize = 0;

} // end constructor

/**
 * ******************* Destructor *******************
 */

template< class TFixedImage, class TMovingImage >
LeigsMetric< TFixedImage, TMovingImage >
::~LeigsMetric()
{
    delete[] this->m_LeigsMetricComputeDerivativePerThreadVariables;
} // end Destructor

/**
 * ************************ SetANNkDTree *************************
 */
    
template< class TFixedImage, class TMovingImage >
void
LeigsMetric< TFixedImage, TMovingImage >
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
LeigsMetric< TFixedImage, TMovingImage >
::SetANNBruteForceTree( )
{
    typename ANNBruteForceTreeType::Pointer tmpPtr = ANNBruteForceTreeType::New();
    
    this->m_BinaryKNNTree  = tmpPtr;
        
} // end SetANNBruteForceTree()

/**
 * ************************ SetANNStandardTreeSearch *************************
 */
    
template< class TFixedImage, class TMovingImage >
void
LeigsMetric< TFixedImage, TMovingImage >
::SetANNStandardTreeSearch( unsigned int kNearestNeighbors, double treeError)
{
    typename ANNStandardTreeSearchType::Pointer tmpPtr = ANNStandardTreeSearchType::New();
        
    tmpPtr->SetKNearestNeighbors( kNearestNeighbors );
        
    tmpPtr->SetErrorBound( treeError );
        
    this->m_BinaryKNNTreeSearcher = tmpPtr;
        
} // end SetANNStandardTreeSearch()

/**
 * ******************* Initialize *******************
 */

template <class TFixedImage, class TMovingImage>
void
LeigsMetric<TFixedImage,TMovingImage>
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
 * ********************* InitializeThreadingParameters ****************************
 */

template< class TFixedImage, class TMovingImage >
void
LeigsMetric< TFixedImage, TMovingImage >
::InitializeThreadingParameters( void ) const
{
    if( this->m_LeigsMetricComputeDerivativePerThreadVariablesSize != this->m_NumberOfThreads )
    {
        delete[] this->m_LeigsMetricComputeDerivativePerThreadVariables;
        this->m_LeigsMetricComputeDerivativePerThreadVariables
        = new AlignedLeigsMetricComputeDerivativePerThreadStruct[ this->m_NumberOfThreads ];
        this->m_LeigsMetricComputeDerivativePerThreadVariablesSize = this->m_NumberOfThreads;
    }
    
    for( ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i )
    {
        this->m_LeigsMetricComputeDerivativePerThreadVariables[ i ].st_Derivative.SetSize( this->GetNumberOfParameters() );
    }
    
} // end InitializeThreadingParameters()


/**
 * ******************* PrintSelf *******************
 */

template < class TFixedImage, class TMovingImage>
void
LeigsMetric<TFixedImage,TMovingImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
    Superclass::PrintSelf( os, indent );

} // end PrintSelf

/**
 * *************** EvaluateTransformJacobianInnerProduct ****************
 */

template < class TFixedImage, class TMovingImage >
void
LeigsMetric<TFixedImage,TMovingImage>
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
LeigsMetric< TFixedImage, TMovingImage >
::ComputeListSamplesLastDim( const ListSamplePointer & listSamplesLastDim ) const
{
    const unsigned int lastDim = this->GetFixedImage()->GetImageDimension() - 1;
    const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );

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
typename LeigsMetric<TFixedImage,TMovingImage>::MeasureType
LeigsMetric<TFixedImage,TMovingImage>
::GetValue( const TransformParametersType & parameters ) const
{
//    itk::TimeProbe timer;
//    timer.Start();

    itkDebugMacro( "GetValue( " << parameters << " ) " );

    const unsigned int lastDim = this->GetFixedImage()->GetImageDimension() - 1;
    const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );
    MeasureType measure = NumericTraits< MeasureType >::Zero;
    
    this->SetTransformParameters( parameters );
    this->GetImageSampler()->Update();

//    timer.Stop();
//    elxout << "Init GV took: "<< static_cast< long >( timer.GetTotal() * 1000 ) << " ms." << std::endl;
//    timer.Start();

    ListSamplePointer listSamplesLastDim  = ListSampleType::New();
    this->ComputeListSamplesLastDim(listSamplesLastDim);
    this->m_BinaryKNNTree->SetSample( listSamplesLastDim );
    this->m_BinaryKNNTree->GenerateTree();
    
    this->m_BinaryKNNTreeSearcher->SetBinaryTree( this->m_BinaryKNNTree );
    
//    timer.Stop();
//    elxout << "Building Graph took: "<< static_cast< long >( timer.GetTotal() * 1000 ) << " ms." << std::endl;
//    timer.Start();

    MeasurementVectorType x,z;
    IndexArrayType        indices;
    DistanceArrayType     distances;
    
    vnl_sparse_matrix<double> allWeights(this->m_NumberOfPixelsCounted,this->m_NumberOfPixelsCounted);
    vnl_sparse_matrix<double> diagonal(this->m_NumberOfPixelsCounted,this->m_NumberOfPixelsCounted);
    vnl_sparse_matrix<double> laplacian(this->m_NumberOfPixelsCounted,this->m_NumberOfPixelsCounted);

    float singleWeight;

    for( unsigned long i = 0; i < this->m_NumberOfPixelsCounted; i++ )
    {
        listSamplesLastDim->GetMeasurementVector(  i, z );
        
        this->m_BinaryKNNTreeSearcher->Search(  z, indices, distances );
        
        for( unsigned long k = 0; k < this->m_NearestNeighbours; k++ )
        {
            listSamplesLastDim->GetMeasurementVector(  indices[k], x );
            singleWeight = std::exp(-vnl_vector_ssd(static_cast<vnl_vector<double> >(z),static_cast<vnl_vector<double> >(x))/this->m_Time);
            allWeights(i,indices[k])= singleWeight;
            allWeights(indices[k],i)= singleWeight;
        }
    }
    
    for( unsigned long k = 0; k < this->m_NumberOfPixelsCounted; k++ )
    {
        for( unsigned long l = 0; l < this->m_NumberOfPixelsCounted; l++ )
        {
            diagonal(k,k) += allWeights(k,l);
        }
    }
    laplacian = (diagonal - allWeights);
    
//    timer.Stop();
//    elxout << "Building matrices took: "<< static_cast< long >( timer.GetTotal() * 1000 ) << " ms." << std::endl;
//    timer.Start();

    vnl_sparse_symmetric_eigensystem eig;
    int error = eig.CalculateNPairs(laplacian, diagonal, 2,0,0,true,true,2000,0);
    
//    timer.Stop();
//    elxout << "Constructing eigenvalues took: "<< static_cast< long >( timer.GetTotal() * 1000 ) << " ms." << std::endl;
//    timer.Start();
    
    return eig.get_eigenvalue(1);

    
} // end GetValue

/**
 * ******************* GetDerivative *******************
 */

template < class TFixedImage, class TMovingImage>
void
LeigsMetric<TFixedImage,TMovingImage>
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
 * ************************ ComputeListSamplesLastDimDerivative *************************
 */
    
template< class TFixedImage, class TMovingImage >
void
LeigsMetric< TFixedImage, TMovingImage >
    ::ComputeListSamplesLastDimDerivative( const ListSamplePointer & listSamplesLastDim, std::vector< FixedImagePointType > * fixedImagePointList) const
{
    const unsigned int lastDim = this->GetFixedImage()->GetImageDimension() - 1;
    const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );
        
    ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
    const unsigned long         nrOfRequestedSamples = sampleContainer->Size();
        
    typename ImageSampleContainerType::ConstIterator fiter;
    typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
    typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();
        
    this->m_NumberOfPixelsCounted = 0;
        
    listSamplesLastDim->SetMeasurementVectorSize(this->m_NumSamplesLastDimension);
    listSamplesLastDim->Resize(nrOfRequestedSamples);
    fixedImagePointList->resize(nrOfRequestedSamples);
    for( fiter = fbegin; fiter != fend; ++fiter )
    {
        FixedImagePointType fixedPoint = ( *fiter ).Value().m_ImageCoordinates;
        
        (*fixedImagePointList)[this->m_NumberOfPixelsCounted]= fixedPoint;
        
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
    fixedImagePointList->resize(this->m_NumberOfPixelsCounted);
}

/**
 * ******************* GetValueAndDerivative *******************
 */
    
template <class TFixedImage, class TMovingImage>
void
LeigsMetric<TFixedImage,TMovingImage>
::GetValueAndDerivative( const TransformParametersType & parameters, MeasureType& value, DerivativeType& derivative ) const
{
    /** Option for now to still use the single threaded code. */
    if( !this->m_UseMultiThread )
    {
        return this->GetValueAndDerivativeSingleThreaded(parameters, value, derivative );
    }
    
    itk::TimeProbe timer;
    timer.Start();
    
    /** Define derivative and Jacobian types. */
    typedef typename DerivativeType::ValueType        DerivativeValueType;
    typedef typename TransformJacobianType::ValueType TransformJacobianValueType;
    
    derivative = DerivativeType( this->GetNumberOfParameters() );
    
    const unsigned int lastDim = this->GetFixedImage()->GetImageDimension() - 1;
    const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );
    
    this->BeforeThreadedGetValueAndDerivative( parameters );
    
    this->InitializeThreadingParameters();

//    timer.Stop();
//    elxout << "Init GVaD took: "<< static_cast< long >( timer.GetTotal() * 1000 ) << " ms." << std::endl;
//    timer.Start();
    
    ListSamplePointer listSamplesLastDim  = ListSampleType::New();
    std::vector<FixedImagePointType> fixedImagePointList;
    this->ComputeListSamplesLastDimDerivative(listSamplesLastDim, &(fixedImagePointList));
    this->m_BinaryKNNTree->SetSample( listSamplesLastDim );
    this->m_BinaryKNNTree->GenerateTree();
    
    this->m_BinaryKNNTreeSearcher->SetBinaryTree( this->m_BinaryKNNTree );
    
//    timer.Stop();
//    elxout << "Building graph took: "<< static_cast< long >( timer.GetTotal() * 1000 ) << " ms." << std::endl;
//    timer.Start();
    
    MeasurementVectorType x,z;
    IndexArrayType        indices;
    DistanceArrayType     distances;
    
    vnl_sparse_matrix<double> allWeights(this->m_NumberOfPixelsCounted,this->m_NumberOfPixelsCounted);
    vnl_sparse_matrix<double> diagonal(this->m_NumberOfPixelsCounted,this->m_NumberOfPixelsCounted);
    vnl_sparse_matrix<double> laplacian(this->m_NumberOfPixelsCounted,this->m_NumberOfPixelsCounted);
    
    float singleWeight;
    
    for( unsigned long i = 0; i < this->m_NumberOfPixelsCounted; i++ )
    {
        listSamplesLastDim->GetMeasurementVector(  i, z );
        
        this->m_BinaryKNNTreeSearcher->Search(  z, indices, distances );
        
        for( unsigned long k = 0; k < this->m_NearestNeighbours; k++ )
        {
            listSamplesLastDim->GetMeasurementVector(  indices[k], x );
            singleWeight = std::exp(-vnl_vector_ssd(static_cast<vnl_vector<double> >(z),static_cast<vnl_vector<double> >(x))/this->m_Time);
            allWeights(i,indices[k])=singleWeight;
            allWeights(indices[k],i)=singleWeight;
        }
    }
    for( unsigned long k = 0; k < this->m_NumberOfPixelsCounted; k++ )
    {
        for( unsigned long l = 0; l < this->m_NumberOfPixelsCounted; l++ )
        {
            diagonal(k,k) += allWeights(k,l);
        }
    }
    
    laplacian = (diagonal - allWeights);
    
//    timer.Stop();
//    elxout << "Building matrices took: "<< static_cast< long >( timer.GetTotal() * 1000 ) << " ms." << std::endl;
//    timer.Start();
    
    vnl_sparse_symmetric_eigensystem eig;
    int error = eig.CalculateNPairs(laplacian, diagonal, 2,0,0,true,true,2000,0);
    
    value = eig.get_eigenvalue(1);
    this->m_FixedImagePointList = &(fixedImagePointList);
    this->m_WeightMatrix = &(allWeights);
    this->m_EigenValue = eig.get_eigenvalue(1);
    this->m_EigenVector = eig.get_eigenvector(1);
//    timer.Stop();
//    elxout << "Constructing eigenvalues took: "<< static_cast< long >( timer.GetTotal() * 1000 ) << " ms." << std::endl;
//    timer.Start();
    
    this->LaunchComputeDerivativeThreaderCallback();

//    timer.Stop();
//    elxout << "Constructing derivatives took: "<< static_cast< long >( timer.GetTotal() * 1000 ) << " ms." << std::endl;
//    timer.Start();

    this->AfterThreadedComputeDerivative( derivative );
    
//    timer.Stop();
//    elxout << "AfterThreaded took: "<< static_cast< long >( timer.GetTotal() * 1000 ) << " ms." << std::endl;
//    timer.Start();


}
    

/**
 * **************** GetSamplesThreaderCallback *******
 */

template< class TFixedImage, class TMovingImage >
ITK_THREAD_RETURN_TYPE
LeigsMetric< TFixedImage, TMovingImage >
::ComputeDerivativeThreaderCallback( void * arg )
{
    ThreadInfoType * infoStruct = static_cast<ThreadInfoType *>( arg );
    ThreadIdType     threadId = infoStruct->ThreadID;
    
    LeigsMetricMultiThreaderParameterType * temp
    = static_cast<LeigsMetricMultiThreaderParameterType *>( infoStruct->UserData );
    
    temp->m_Metric->ThreadedComputeDerivative( threadId );
    
    return ITK_THREAD_RETURN_VALUE;
    
} // GetSamplesThreaderCallback()


/**
 * *********************** LaunchGetSamplesThreaderCallback***************
 */

template< class TFixedImage, class TMovingImage >
void
LeigsMetric< TFixedImage, TMovingImage >
::LaunchComputeDerivativeThreaderCallback( void ) const
{
    /** Setup local threader. */
    // \todo: is a global threader better performance-wise? check
    typename ThreaderType::Pointer local_threader = ThreaderType::New();
    local_threader->SetNumberOfThreads( this->m_NumberOfThreads );
    local_threader->SetSingleMethod( this->ComputeDerivativeThreaderCallback,const_cast<void *>( static_cast<const void *>(&this->m_LeigsMetricThreaderParameters ) ) );
    
    /** Launch. */
    local_threader->SingleMethodExecute();
    
} // end LaunchGetSamplesThreaderCallback()

/**
 * ******************* GetValueAndDerivativeSingleThreaded *******************
 */

template <class TFixedImage, class TMovingImage>
void
LeigsMetric<TFixedImage,TMovingImage>
::GetValueAndDerivativeSingleThreaded( const TransformParametersType & parameters, MeasureType& value, DerivativeType& derivative ) const
{
    itkDebugMacro("GetValueAndDerivative( " << parameters << " ) ");
    
    itk::TimeProbe timer;
    timer.Start();

    /** Define derivative and Jacobian types. */
    typedef typename DerivativeType::ValueType        DerivativeValueType;
    typedef typename TransformJacobianType::ValueType TransformJacobianValueType;
    
    derivative = DerivativeType( this->GetNumberOfParameters() );
    derivative.Fill(0.0);

    const unsigned int lastDim = this->GetFixedImage()->GetImageDimension() - 1;
    const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );
    
    this->BeforeThreadedGetValueAndDerivative( parameters );
    
//    timer.Stop();
//    elxout << "Init GVaD took: "<< static_cast< long >( timer.GetTotal() * 1000 ) << " ms." << std::endl;
//    timer.Start();

    ListSamplePointer listSamplesLastDim  = ListSampleType::New();
    std::vector<FixedImagePointType> fixedImagePointList;
    this->ComputeListSamplesLastDimDerivative(listSamplesLastDim, &(fixedImagePointList));
    this->m_BinaryKNNTree->SetSample( listSamplesLastDim );
    this->m_BinaryKNNTree->GenerateTree();
    
    this->m_BinaryKNNTreeSearcher->SetBinaryTree( this->m_BinaryKNNTree );
    
//    timer.Stop();
//    elxout << "Building graph took: "<< static_cast< long >( timer.GetTotal() * 1000 ) << " ms." << std::endl;
//    timer.Start();

    MeasurementVectorType x,z;
    IndexArrayType        indices;
    DistanceArrayType     distances;
    
    vnl_sparse_matrix<double> allWeights(this->m_NumberOfPixelsCounted,this->m_NumberOfPixelsCounted);
    vnl_sparse_matrix<double> diagonal(this->m_NumberOfPixelsCounted,this->m_NumberOfPixelsCounted);
    vnl_sparse_matrix<double> laplacian(this->m_NumberOfPixelsCounted,this->m_NumberOfPixelsCounted);

    float singleWeight;
    
    for( unsigned long i = 0; i < this->m_NumberOfPixelsCounted; i++ )
    {
        listSamplesLastDim->GetMeasurementVector(  i, z );
        
        this->m_BinaryKNNTreeSearcher->Search(  z, indices, distances );
        
        for( unsigned long k = 0; k < this->m_NearestNeighbours; k++ )
        {
            listSamplesLastDim->GetMeasurementVector(  indices[k], x );
            singleWeight = std::exp(-vnl_vector_ssd(static_cast<vnl_vector<double> >(z),static_cast<vnl_vector<double> >(x))/this->m_Time);
            allWeights(i,indices[k])=singleWeight;
            allWeights(indices[k],i)=singleWeight;
        }
    }
    for( unsigned long k = 0; k < this->m_NumberOfPixelsCounted; k++ )
    {
        for( unsigned long l = 0; l < this->m_NumberOfPixelsCounted; l++ )
        {
            diagonal(k,k) += allWeights(k,l);
        }
    }
    
    laplacian = (diagonal - allWeights);

//    timer.Stop();
//    elxout << "Building matrices took: "<< static_cast< long >( timer.GetTotal() * 1000 ) << " ms." << std::endl;
//    timer.Start();

    vnl_sparse_symmetric_eigensystem eig;
    int error = eig.CalculateNPairs(laplacian, diagonal, 2,0,0,true,true,2000,0);

    value = eig.get_eigenvalue(1);
//    timer.Stop();
//    elxout << "Constructing eigenvalues took: "<< static_cast< long >( timer.GetTotal() * 1000 ) << " ms." << std::endl;
//    timer.Start();

    float weight,singleDerWeight;
    for( unsigned long i = 0; i < this->m_NumberOfPixelsCounted; i++ )
    {
        FixedImageContinuousIndexType voxelCoord,voxelCoord_j;
        this->GetFixedImage()->TransformPhysicalPointToContinuousIndex( fixedImagePointList[i], voxelCoord );

        TransformJacobianType  jacobianTemp;
        
        NonZeroJacobianIndicesType nzjiTemp( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices());
        
        DerivativeType imageJacobianTemp( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
        for( unsigned int s = 0; s < this->m_NumSamplesLastDimension; s++ )
        {
            MovingImagePointType                    mappedPoint;
            RealType                                movingImageValueTemp,movingImageValueTemp_j;
            MovingImageDerivativeType               movingImageDerivativeTemp;
            FixedImagePointType                     fixedPoint;

            voxelCoord[ lastDim ] = this->m_RandomList[ s ];

            this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord, fixedPoint );
            this->TransformPoint( fixedPoint, mappedPoint );
            this->EvaluateMovingImageValueAndDerivative(mappedPoint, movingImageValueTemp, &movingImageDerivativeTemp );
            this->EvaluateTransformJacobian( fixedPoint, jacobianTemp, nzjiTemp);
            this->EvaluateTransformJacobianInnerProduct(jacobianTemp, movingImageDerivativeTemp, imageJacobianTemp );

            vnl_matrix<double> derWeights(this->m_NumberOfPixelsCounted,this->m_NumberOfPixelsCounted,0);
            vnl_diag_matrix<double> derDiagonal(this->m_NumberOfPixelsCounted,0);
            
            for( unsigned long j = 0; j < this->m_NumberOfPixelsCounted; j++ )
            {
                if(allWeights(i,j) != 0)
                {
                    this->GetFixedImage()->TransformPhysicalPointToContinuousIndex( fixedImagePointList[j], voxelCoord_j );

                    voxelCoord_j[ lastDim ] = this->m_RandomList[ s ];
                    
                    this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord_j, fixedPoint );
                    this->TransformPoint( fixedPoint, mappedPoint );
                    this->EvaluateMovingImageValueAndDerivative(mappedPoint, movingImageValueTemp_j, 0 );
                    singleDerWeight = - 2.0 / this->m_Time * allWeights(i,j) * (movingImageValueTemp - movingImageValueTemp_j);
                    derWeights(i,j) = singleDerWeight;
                    derWeights(j,i) = singleDerWeight;
                    derDiagonal(j,j) = singleDerWeight;
                    derDiagonal(i,i) += singleDerWeight;

                }
            }

//            timer.Stop();
//            elxout << "Building matrices for derivatives for one sample took: "<< static_cast< long >( timer.GetTotal() * 1000 ) << " ms." << std::endl;
//            timer.Start();

            derDiagonal.set_diagonal(derWeights.get_diagonal()*(1-value));
            derWeights = derDiagonal - derWeights;
            
//            timer.Stop();
//            elxout << "Building final matrix for derivatives for one sample took: "<< static_cast< long >( timer.GetTotal() * 1000 ) << " ms." << std::endl;
//            timer.Start();
            
            weight = dot_product(eig.get_eigenvector(1)*derWeights,eig.get_eigenvector(1));
            
//            timer.Stop();
//            elxout << "Calculating the actual weight took: "<< static_cast< long >( timer.GetTotal() * 1000 ) << " ms." << std::endl;
//            timer.Start();

            for( unsigned int p = 0; p < nzjiTemp.size(); ++p)
            {
                derivative[ nzjiTemp[p] ] +=  weight*imageJacobianTemp[p];
            }
            
//            timer.Stop();
//            elxout << "Applying the weight to the ImageJacobian took: "<< static_cast< long >( timer.GetTotal() * 1000 ) << " ms." << std::endl;
//            timer.Start();

            
        }
//        timer.Stop();
//        elxout << "Calculating derivatives for one sample took: "<< static_cast< long >( timer.GetTotal() * 1000 ) << " ms." << std::endl;
//        timer.Start();


    }
//    timer.Stop();
//    elxout << "Calculating derivatives took: "<< static_cast< long >( timer.GetTotal() * 1000 ) << " ms." << std::endl;
//    timer.Start();

    if( this->m_SubtractMean )
    {
        if( ! this->m_TransformIsStackTransform )
        {
            
        }
        else
        {
            const unsigned int numParametersPerLastDimension = this->GetNumberOfParameters() / lastDimSize;
            DerivativeType mean ( numParametersPerLastDimension );
            mean.Fill( 0.0 );
            
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
//    timer.Stop();
//    elxout << "Subtracting derivatives took: "<< static_cast< long >( timer.GetTotal() * 1000 ) << " ms." << std::endl;

} // end GetValueAndDerivative()

/**
 * ******************* ThreadedComputeDerivative *******************
 */

template< class TFixedImage, class TMovingImage >
void
LeigsMetric< TFixedImage, TMovingImage >
::ThreadedComputeDerivative( ThreadIdType threadId )
{
    
    DerivativeType & derivative = this->m_LeigsMetricComputeDerivativePerThreadVariables[ threadId ].st_Derivative;
    derivative.Fill( 0.0 );

    const unsigned int lastDim = this->GetFixedImage()->GetImageDimension() - 1;
    const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );
    
    const unsigned long nrOfSamplesPerThreads = static_cast< unsigned long >( vcl_ceil( static_cast< double >( this->m_NumberOfPixelsCounted ) / static_cast< double >( this->m_NumberOfThreads ) ) );

    unsigned long pos_begin = nrOfSamplesPerThreads * threadId;
    unsigned long pos_end   = nrOfSamplesPerThreads * ( threadId + 1 );
    pos_begin = ( pos_begin > this->m_NumberOfPixelsCounted ) ? this->m_NumberOfPixelsCounted : pos_begin;
    pos_end   = ( pos_end > this->m_NumberOfPixelsCounted ) ? this->m_NumberOfPixelsCounted : pos_end;
    
    float weight,singleDerWeight;
    for( unsigned long i = pos_begin; i < pos_end; i++ )
    {
        FixedImageContinuousIndexType voxelCoord,voxelCoord_j;
        this->GetFixedImage()->TransformPhysicalPointToContinuousIndex( (*this->m_FixedImagePointList)[i], voxelCoord );

        TransformJacobianType  jacobianTemp;
        
        NonZeroJacobianIndicesType nzjiTemp( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices());

        DerivativeType imageJacobianTemp( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
        for( unsigned int s = 0; s < this->m_NumSamplesLastDimension; s++ )
        {
            MovingImagePointType                    mappedPoint;
            RealType                                movingImageValueTemp,movingImageValueTemp_j;
            MovingImageDerivativeType               movingImageDerivativeTemp;
            FixedImagePointType                     fixedPoint;

            voxelCoord[ lastDim ] = this->m_RandomList[ s ];
            
            this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord, fixedPoint );
            this->TransformPoint( fixedPoint, mappedPoint );
            this->EvaluateMovingImageValueAndDerivative(mappedPoint, movingImageValueTemp, &movingImageDerivativeTemp );
            this->EvaluateTransformJacobian( fixedPoint, jacobianTemp, nzjiTemp);
            this->EvaluateTransformJacobianInnerProduct(jacobianTemp, movingImageDerivativeTemp, imageJacobianTemp );

            vnl_matrix<double> derWeights(this->m_NumberOfPixelsCounted,this->m_NumberOfPixelsCounted,0);
            vnl_diag_matrix<double> derDiagonal(this->m_NumberOfPixelsCounted,0);
            
            for( unsigned long j = 0; j < this->m_NumberOfPixelsCounted; j++ )
            {
                if((*this->m_WeightMatrix)(i,j) != 0)
                {
                    this->GetFixedImage()->TransformPhysicalPointToContinuousIndex( (*this->m_FixedImagePointList)[j], voxelCoord_j );
                    
                    voxelCoord_j[ lastDim ] = this->m_RandomList[ s ];
                    
                    this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord_j, fixedPoint );
                    this->TransformPoint( fixedPoint, mappedPoint );
                    this->EvaluateMovingImageValueAndDerivative(mappedPoint, movingImageValueTemp_j, 0 );
                    singleDerWeight = - 2.0 / this->m_Time * (*this->m_WeightMatrix)(i,j) * (movingImageValueTemp - movingImageValueTemp_j);
                    derWeights(i,j) = singleDerWeight;
                    derWeights(j,i) = singleDerWeight;
                    derDiagonal(j,j) = singleDerWeight;
                    derDiagonal(i,i) += singleDerWeight;
                    
                }
            }

            derDiagonal.set_diagonal(derWeights.get_diagonal()*(1-this->m_EigenValue));
            derWeights = derDiagonal - derWeights;

            weight = dot_product(this->m_EigenVector*derWeights,this->m_EigenVector);

            for( unsigned int p = 0; p < nzjiTemp.size(); ++p)
            {
                derivative[ nzjiTemp[p] ] +=  weight*imageJacobianTemp[p];
            }
        }
    }
}
    
/**
 * ******************* AfterThreadedComputeDerivative *******************
 */

template< class TFixedImage, class TMovingImage >
void
LeigsMetric< TFixedImage, TMovingImage >
::AfterThreadedComputeDerivative(DerivativeType & derivative ) const
{
    derivative = this->m_LeigsMetricComputeDerivativePerThreadVariables[ 0 ].st_Derivative;
    for( ThreadIdType i = 1; i < this->m_NumberOfThreads; ++i )
    {
        derivative += this->m_LeigsMetricComputeDerivativePerThreadVariables[ i ].st_Derivative;
    }
    const unsigned int lastDim = this->GetFixedImage()->GetImageDimension() - 1;
    const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );

    /** Subtract mean from derivative elements. */
    if( this->m_SubtractMean )
    {
        if( !this->m_TransformIsStackTransform )
        {
        }
        else
        {
            /** Update derivative per dimension.
             * Parameters are ordered x0x0x0y0y0y0z0z0z0x1x1x1y1y1y1z1z1z1 with
             * the number the time point index.
             */
            const unsigned int numParametersPerLastDimension = this->GetNumberOfParameters() / lastDimSize;
            DerivativeType mean( numParametersPerLastDimension );
            mean.Fill( 0.0 );
            
            /** Compute mean per control point. */
            for( unsigned int t = 0; t < lastDimSize; ++t )
            {
                const unsigned int startc = numParametersPerLastDimension * t;
                for( unsigned int c = startc; c < startc + numParametersPerLastDimension; ++c )
                {
                    const unsigned int index = c % numParametersPerLastDimension;
                    mean[ index ] += derivative[ c ];
                }
            }
            mean /= static_cast<RealType>( lastDimSize );
            
            /** Update derivative per control point. */
            for( unsigned int t = 0; t < lastDimSize; ++t )
            {
                const unsigned int startc = numParametersPerLastDimension * t;
                for( unsigned int c = startc; c < startc + numParametersPerLastDimension; ++c )
                {
                    const unsigned int index = c % numParametersPerLastDimension;
                    derivative[ c ] -= mean[ index ];
                }
            }
        }
    }
}
    
/**
 * ******************* SampleRandom *******************
 */
    
template< class TFixedImage, class TMovingImage >
void
LeigsMetric< TFixedImage, TMovingImage >
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

#endif // ITKLeigsMetric_HXX
