#ifndef _itkLinearGroupwiseMSD_HXX__
#define _itkLinearGroupwiseMSD_HXX__

#include "itkLinearGroupwiseMSD.h"

#ifdef ELASTIX_USE_OPENMP
#include <omp.h>
#endif

namespace itk
{
    /**
     * ********************* Constructor ******************************
     */
    
    template< class TFixedImage, class TMovingImage >
    LinearGroupwiseMSD< TFixedImage, TMovingImage >
    ::LinearGroupwiseMSD()
    {
        this->SetUseImageSampler( true );
        this->SetUseFixedImageLimiter( false );
        this->SetUseMovingImageLimiter( false );
        
        this->m_LinearGroupwiseMSDThreaderParameters.m_Metric = this;
        
        this->m_LinearGroupwiseMSDGetValueAndDerivativePerThreadVariables     = NULL;
        this->m_LinearGroupwiseMSDGetValueAndDerivativePerThreadVariablesSize = 0;

 
    }
    
    /**
     * ******************* Destructor *******************
     */
    
    template< class TFixedImage, class TMovingImage >
    LinearGroupwiseMSD< TFixedImage, TMovingImage >
    ::~LinearGroupwiseMSD()
    {
        delete[] this->m_LinearGroupwiseMSDGetValueAndDerivativePerThreadVariables;
    }
    
    /**
     * ******************* Initialize *******************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearGroupwiseMSD< TFixedImage, TMovingImage >
    ::Initialize( void ) throw ( ExceptionObject )
    {
        this->Superclass::Initialize();
        
        const unsigned int lastDim     = this->GetFixedImage()->GetImageDimension() - 1;
        const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );

        if( (this->m_NumSamplesLastDimension > lastDimSize) || (!this->m_SampleLastDimensionRandomly) )
        {
            this->m_NumSamplesLastDimension = lastDimSize;
        }
        
        
        ImageLinearConstIteratorWithIndex< MovingImageType > it( this->GetMovingImage(), this->GetMovingImage()->GetLargestPossibleRegion() );
        it.SetDirection( lastDim );
        it.GoToBegin();
        
        double sumvar = 0.0;
        int   num    = 0;
        while( !it.IsAtEnd() )
        {
            double        sum     = 0.0;
            double        sumsq   = 0.0;
            unsigned int numlast = 0;
            while( !it.IsAtEndOfLine() )
            {
                double value = it.Get();
                sum   += value;
                sumsq += value * value;
                ++numlast;
                ++it;
            }
            
            double expectedValue = sum / static_cast< double >( numlast );
            sumvar += sumsq / static_cast< double >( numlast ) - expectedValue * expectedValue;
            num++;
            
            it.NextLine();
        }
                
        if( sumvar == 0 )
        {
            this->m_InitialVariance = 1.0f;
        }
        else
        {
            this->m_InitialVariance = sumvar / static_cast< double >( num );
        }
        
        std::vector<double> intensityConstants(lastDimSize);
        for(unsigned int d = 0; d < lastDimSize; d++)
        {
            FixedImagePixelType trueMinTemp = NumericTraits< FixedImagePixelType >::max();
            
            /** If no mask. */
            if( this->m_FixedImageMask.IsNull() )
            {
                typedef ImageRegionConstIterator< FixedImageType > IteratorType;
                
                FixedImageRegionType region = this->GetFixedImage()->GetLargestPossibleRegion();
                region.SetIndex(ReducedFixedImageDimension, d);
                region.SetSize(ReducedFixedImageDimension, 1);
                IteratorType it(this->GetFixedImage(), region );
                
                for( it.GoToBegin(); !it.IsAtEnd(); ++it )
                {
                    const FixedImagePixelType sample = it.Get();
                    trueMinTemp = vnl_math_min( trueMinTemp, sample );
                    
                }
            }
            
            else
            {
                typedef ImageRegionConstIteratorWithIndex< FixedImageType > IteratorType;
                
                FixedImageRegionType region = this->GetFixedImage()->GetLargestPossibleRegion();
                
                region.SetIndex(ReducedFixedImageDimension, d);
                region.SetSize(ReducedFixedImageDimension, 1);
                
                IteratorType it(this->GetFixedImage(), region );
                
                for( it.GoToBegin(); !it.IsAtEnd(); ++it )
                {
                    OutputPointType point;
                    this->GetFixedImage()->TransformIndexToPhysicalPoint( it.GetIndex(), point );
                    if( this->m_FixedImageMask->IsInside( point ) )
                    {
                        const FixedImagePixelType sample = it.Get();
                        trueMinTemp = vnl_math_min( trueMinTemp, sample );
                        
                    }
                }
            }
            
            if( trueMinTemp < 1)
            {
                intensityConstants[d] = 0 - trueMinTemp;
            }
        }
        
        this->m_TemplateImage->SetIntensityConstants(intensityConstants);

    }
    
    
    /**
     * ******************* GetValue *******************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearGroupwiseMSD< TFixedImage, TMovingImage >
    ::GetValue(const ParametersType & parameters, std::vector<MeasureType> & values, bool & minimize) const
    {
        
        this->m_NumberOfPixelsCounted = 0;
        values.assign(this->m_NumSamplesLastDimension,0.0);

        const unsigned int lastDim     = this->GetFixedImage()->GetImageDimension() - 1;
        const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );

        this->BeforeThreadedGetValueAndDerivative( parameters );
        
        ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
        
        typename ImageSampleContainerType::ConstIterator fiter;
        typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
        typename ImageSampleContainerType::ConstIterator fend   = sampleContainer->End();
        
        std::vector< int > lastDimPositions(this->m_NumSamplesLastDimension);
        std::vector< unsigned long> NumberOfPixelsCountedVector(lastDimSize);

        if( !this->m_SampleLastDimensionRandomly )
        {
            for( unsigned int d = 0; d < lastDimSize; d++ )
            {
                lastDimPositions[d] = d;
            }
        }
        else
        {
            this->SampleRandom( this->m_NumSamplesLastDimension, lastDimSize, lastDimPositions );
        }
        
        for( fiter = fbegin; fiter != fend; ++fiter )
        {
            FixedImagePointType fixedPoint = ( *fiter ).Value().m_ImageCoordinates;
            
            
            FixedImageContinuousIndexType voxelCoord;
            this->GetFixedImage()->TransformPhysicalPointToContinuousIndex( fixedPoint, voxelCoord );
            
            unsigned int       numSamplesOk            = 0;
            std::vector< double >  imageValues(this->m_NumSamplesLastDimension);
            std::vector< unsigned int > positions(this->m_NumSamplesLastDimension);

            for( unsigned int s = 0; s < this->m_NumSamplesLastDimension; s++ )
            {
                RealType             movingImageValue;
                MovingImagePointType mappedPoint;
                
                voxelCoord[ lastDim ] = lastDimPositions[ s ];
                
                this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord, fixedPoint );
                
                bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );
                
                if( sampleOk )
                {
                    sampleOk = this->IsInsideMovingMask( mappedPoint );
                }
                
                if( sampleOk )
                {
                    sampleOk = this->EvaluateMovingImageValueAndDerivative(mappedPoint, movingImageValue, 0 );
                }
                
                if( sampleOk )
                {
                    imageValues[numSamplesOk] = movingImageValue;
                    positions[numSamplesOk] = lastDimPositions[ s ];
                    numSamplesOk++;
                }

            }

            if( numSamplesOk > 1 )
            {
                this->m_NumberOfPixelsCounted++;
                double templateIntensity= this->m_TemplateImage->CalculateIntensity(imageValues, numSamplesOk, positions);
                for( unsigned int o = 0; o < numSamplesOk; o++ )
                {
                    NumberOfPixelsCountedVector[positions[o]]++;
                    values[positions[o]] += (this->m_TemplateImage->TransformIntensity(imageValues[o], positions[o])-templateIntensity)*(this->m_TemplateImage->TransformIntensity(imageValues[o], positions[o])-templateIntensity);
                }
                
            }
        }

        for( unsigned int d = 0; d < lastDimSize; d++ )
        {
            values[d] /= (this->m_InitialVariance * NumberOfPixelsCountedVector[d]);
        }
        
        minimize = true;
        
    }
    
    
    /**
     * ******************* GetValueAndDerivative *******************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearGroupwiseMSD< TFixedImage, TMovingImage >
    ::GetValueAndDerivative( const ParametersType & parameters, std::vector<MeasureType> & values, std::vector<DerivativeType> & derivatives, bool & minimize) const
    {
        if( !this->m_UseMultiThread )
        {
            return this->GetValueAndDerivativeSingleThreaded(parameters, values, derivatives );
        }
        
        this->BeforeThreadedGetValueAndDerivative( parameters );
        this->InitializeThreadingParameters();
        this->LaunchGetValueAndDerivativeThreaderCallback();
        this->AfterThreadedGetValueAndDerivative( values, derivatives );

        minimize = true;
    }
    
    /**
     * ******************* GetValueAndDerivativeSingleThreaded *******************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearGroupwiseMSD< TFixedImage, TMovingImage >
    ::GetValueAndDerivativeSingleThreaded( const ParametersType & parameters, std::vector<MeasureType> & values, std::vector<DerivativeType> & derivatives ) const
    {
        
        const unsigned int lastDim     = this->GetFixedImage()->GetImageDimension() - 1;
        const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );
        
        values.assign(this->m_NumSamplesLastDimension,0.0);
        derivatives.resize(lastDimSize);
        for(unsigned int d= 0; d < lastDimSize; d++)
        {
            derivatives[d] = DerivativeType( this->GetNumberOfParameters() );
            derivatives[d].Fill(NumericTraits< DerivativeValueType >::Zero);
        }
        
        this->m_NumberOfPixelsCounted = 0;
        std::vector< unsigned long> NumberOfPixelsCountedVector(lastDimSize);
        
        TransformJacobianType jacobian;
        DerivativeType        imageJacobian( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
        DerivativeType dMTAdmu( this->GetNumberOfParameters() );
        
        std::vector< double >  imageValues(this->m_NumSamplesLastDimension);
        std::vector< unsigned int > positions(this->m_NumSamplesLastDimension);
        std::vector< DerivativeType > imageJacobians( this->m_NumSamplesLastDimension );
        std::vector< NonZeroJacobianIndicesType > nzjis(this->m_NumSamplesLastDimension, NonZeroJacobianIndicesType() );
        
        std::vector< int > lastDimPositions(this->m_NumSamplesLastDimension);
        if( !this->m_SampleLastDimensionRandomly )
        {
            for( unsigned int d = 0; d < lastDimSize; d++ )
            {
                lastDimPositions[d] = d;
            }
        }
        else
        {
            this->SampleRandom( this->m_NumSamplesLastDimension, lastDimSize, lastDimPositions );
        }
        
        this->BeforeThreadedGetValueAndDerivative( parameters );
        
        ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
        
        typename ImageSampleContainerType::ConstIterator fiter;
        typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
        typename ImageSampleContainerType::ConstIterator fend   = sampleContainer->End();

        for( fiter = fbegin; fiter != fend; ++fiter )
        {
            FixedImagePointType fixedPoint = ( *fiter ).Value().m_ImageCoordinates;
            
            FixedImageContinuousIndexType voxelCoord;
            this->GetFixedImage()->TransformPhysicalPointToContinuousIndex( fixedPoint, voxelCoord );
            
            unsigned int       numSamplesOk            = 0;
            
            for( unsigned int s = 0; s < this->m_NumSamplesLastDimension; s++ )
            {
                RealType             movingImageValue;
                MovingImagePointType mappedPoint;
                MovingImageDerivativeType movingImageDerivative;
                
                voxelCoord[ lastDim ] = lastDimPositions[ s ];
                
                this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord, fixedPoint );
                
                bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );
                
                if( sampleOk )
                {
                    sampleOk = this->IsInsideMovingMask( mappedPoint );
                }
                
                if( sampleOk )
                {
                    sampleOk = this->EvaluateMovingImageValueAndDerivative(mappedPoint, movingImageValue, &movingImageDerivative );
                }
                
                if( sampleOk )
                {
                    this->EvaluateTransformJacobian( fixedPoint, jacobian, nzjis[ numSamplesOk ] );
                    this->EvaluateTransformJacobianInnerProduct(jacobian, movingImageDerivative, imageJacobian );

                    imageValues[numSamplesOk] = movingImageValue;
                    imageJacobians[numSamplesOk] = imageJacobian;
                    positions[numSamplesOk] = lastDimPositions[ s ];
                    numSamplesOk++;
                }
                
            }
            
            if( numSamplesOk > 1 )
            {
                this->m_NumberOfPixelsCounted++;
                double templateIntensity=this->m_TemplateImage->CalculateIntensity(imageValues, numSamplesOk, positions);

                for( unsigned int o = 0; o < numSamplesOk; o++ )
                {
                    for( unsigned int j = 0; j < nzjis[ o ].size(); ++j )
                    {
                        dMTAdmu[nzjis[ o ][ j ]] = 0.0;
                    }
                }
                
                for( unsigned int o = 0; o < numSamplesOk; o++ )
                {
                    typename DerivativeType::const_iterator imJac = imageJacobians[o].begin();
                    for( unsigned int j = 0; j < nzjis[ o ].size(); j++ )
                    {
                        dMTAdmu[nzjis[ o ][ j ]] += this->m_TemplateImage->CalculateRatio(imageValues[o], templateIntensity,positions[o]) * (*imJac) / static_cast< float >( numSamplesOk );
                        ++imJac;
                    }
                }
                
                for( unsigned int o = 0; o < numSamplesOk; o++ )
                {
                    NumberOfPixelsCountedVector[positions[o]]++;
                    //hier heb ik / static_cast< float >( numSamplesOk ) verwijdert
                    values[positions[o]] += (this->m_TemplateImage->TransformIntensity(imageValues[o], positions[o])-templateIntensity)*(this->m_TemplateImage->TransformIntensity(imageValues[o], positions[o])-templateIntensity);
                    
                    typename DerivativeType::const_iterator imJac = imageJacobians[o].begin();
                    for( unsigned int j = 0; j < nzjis[ o ].size(); ++j )
                    {
                        //hier heb ik / static_cast< float >( numSamplesOk ) verwijdert
                        derivatives[positions[o]][ nzjis[ o ][ j ] ] += ( 2.0 * (this->m_TemplateImage->TransformIntensity(imageValues[o], positions[o]) - templateIntensity ) * (*imJac - dMTAdmu[ nzjis[ o ][ j ] ]));
                        ++imJac;
                    }
                }
            }
        }
        
        this->CheckNumberOfSamples(sampleContainer->Size(), this->m_NumberOfPixelsCounted );

        for( unsigned int d = 0; d < lastDimSize; d++ )
        {
            values[d] /= (this->m_InitialVariance * NumberOfPixelsCountedVector[d]);
            derivatives[d] /= (this->m_InitialVariance * NumberOfPixelsCountedVector[d]);
        }
        
    }
    
    /**
     * ******************* InitializeThreadingParameters *******************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearGroupwiseMSD< TFixedImage, TMovingImage >
    ::InitializeThreadingParameters(void) const
    {
        Superclass::InitializeThreadingParameters();
        
        const unsigned int lastDim     = this->GetFixedImage()->GetImageDimension() - 1;
        const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );
        
        if( this->m_LinearGroupwiseMSDGetValueAndDerivativePerThreadVariablesSize != this->m_NumberOfThreads )
        {
            delete[] this->m_LinearGroupwiseMSDGetValueAndDerivativePerThreadVariables;
            this->m_LinearGroupwiseMSDGetValueAndDerivativePerThreadVariables
            = new AlignedLinearGroupwiseMSDGetValueAndDerivativePerThreadStruct[this->m_NumberOfThreads ];
            this->m_LinearGroupwiseMSDGetValueAndDerivativePerThreadVariablesSize = this->m_NumberOfThreads;
        }

        for( ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i )
        {
            this->m_LinearGroupwiseMSDGetValueAndDerivativePerThreadVariables[ i ].st_NumberOfPixelsCounted = NumericTraits< SizeValueType >::Zero;
            this->m_LinearGroupwiseMSDGetValueAndDerivativePerThreadVariables[ i ].st_NumberOfPixelsCountedVector.assign(lastDimSize, NumericTraits< SizeValueType >::Zero);
            this->m_LinearGroupwiseMSDGetValueAndDerivativePerThreadVariables[ i ].st_Values.assign(lastDimSize,0.0);
            this->m_LinearGroupwiseMSDGetValueAndDerivativePerThreadVariables[ i ].st_Derivatives.resize(lastDimSize);
        }

        this->m_RandomList.assign(this->m_NumSamplesLastDimension,0.0);

        if( !this->m_SampleLastDimensionRandomly )
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
 
    }
    
    
    /**
     * ******************* ThreadedGetValueAndDerivative *******************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearGroupwiseMSD< TFixedImage, TMovingImage >
    ::ThreadedGetValueAndDerivative( ThreadIdType threadId )
    {
        const unsigned int lastDim     = this->GetFixedImage()->GetImageDimension() - 1;
        const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );

        std::vector<double> & values = this->m_LinearGroupwiseMSDGetValueAndDerivativePerThreadVariables[ threadId ].st_Values;
        std::vector<DerivativeType> & derivatives = this->m_LinearGroupwiseMSDGetValueAndDerivativePerThreadVariables[ threadId ].st_Derivatives;
        std::vector< unsigned long> & numberOfPixelsCountedVector = this->m_LinearGroupwiseMSDGetValueAndDerivativePerThreadVariables[ threadId ].st_NumberOfPixelsCountedVector;

        for(unsigned int d= 0; d < lastDimSize; d++)
        {
            derivatives[d] = DerivativeType( this->GetNumberOfParameters() );
            derivatives[d].Fill(NumericTraits< DerivativeValueType >::Zero);
        }
        
        unsigned long numberOfPixelsCounted = 0;
        
        TransformJacobianType jacobian;
        DerivativeType        imageJacobian( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
        DerivativeType dMTAdmu( this->GetNumberOfParameters() );
        
        std::vector< double >  imageValues(this->m_NumSamplesLastDimension);
        std::vector< unsigned int > positions(this->m_NumSamplesLastDimension);
        std::vector< DerivativeType > imageJacobians( this->m_NumSamplesLastDimension );
        std::vector< NonZeroJacobianIndicesType > nzjis(this->m_NumSamplesLastDimension, NonZeroJacobianIndicesType() );
        
        ImageSampleContainerPointer sampleContainer     = this->GetImageSampler()->GetOutput();
        const unsigned long         sampleContainerSize = sampleContainer->Size();
        
        const unsigned long nrOfSamplesPerThreads = static_cast< unsigned long >( vcl_ceil( static_cast< double >( sampleContainerSize ) / static_cast< double >( this->m_NumberOfThreads ) ) );
        
        unsigned long pos_begin = nrOfSamplesPerThreads * threadId;
        unsigned long pos_end   = nrOfSamplesPerThreads * ( threadId + 1 );
        pos_begin = ( pos_begin > sampleContainerSize ) ? sampleContainerSize : pos_begin;
        pos_end   = ( pos_end > sampleContainerSize ) ? sampleContainerSize : pos_end;
        
        typename ImageSampleContainerType::ConstIterator fiter;
        typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
        typename ImageSampleContainerType::ConstIterator fend   = sampleContainer->Begin();
        fbegin                                                 += (int)pos_begin;
        fend                                                   += (int)pos_end;
        
        for( fiter = fbegin; fiter != fend; ++fiter )
        {
            FixedImagePointType fixedPoint = ( *fiter ).Value().m_ImageCoordinates;
            
            FixedImageContinuousIndexType voxelCoord;
            this->GetFixedImage()->TransformPhysicalPointToContinuousIndex( fixedPoint, voxelCoord );
            
            unsigned int       numSamplesOk            = 0;
            
            for( unsigned int s = 0; s < this->m_NumSamplesLastDimension; s++ )
            {
                RealType             movingImageValue;
                MovingImagePointType mappedPoint;
                MovingImageDerivativeType movingImageDerivative;
                
                voxelCoord[ lastDim ] = this->m_RandomList[ s ];
                
                this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord, fixedPoint );
                
                bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );
                
                if( sampleOk )
                {
                    sampleOk = this->IsInsideMovingMask( mappedPoint );
                }
                
                if( sampleOk )
                {
                    sampleOk = this->EvaluateMovingImageValueAndDerivative(mappedPoint, movingImageValue, &movingImageDerivative );
                }
                
                if( sampleOk )
                {
                    this->EvaluateTransformJacobian( fixedPoint, jacobian, nzjis[ numSamplesOk ] );
                    this->EvaluateTransformJacobianInnerProduct(jacobian, movingImageDerivative, imageJacobian );
                    
                    imageValues[numSamplesOk] = movingImageValue;
                    imageJacobians[numSamplesOk] = imageJacobian;
                    positions[numSamplesOk] = this->m_RandomList[ s ];
                    numSamplesOk++;
                }
                
            }
            
            if( numSamplesOk > 1 )
            {
                numberOfPixelsCounted++;
                double templateIntensity=this->m_TemplateImage->CalculateIntensity(imageValues, numSamplesOk, positions);
                
                for( unsigned int o = 0; o < numSamplesOk; o++ )
                {
                    for( unsigned int j = 0; j < nzjis[ o ].size(); ++j )
                    {
                        dMTAdmu[nzjis[ o ][ j ]] = 0.0;
                    }
                }
                
                for( unsigned int o = 0; o < numSamplesOk; o++ )
                {
                    numberOfPixelsCountedVector[positions[o]]++;
                    typename DerivativeType::const_iterator imJac = imageJacobians[o].begin();
                    for( unsigned int j = 0; j < nzjis[ o ].size(); j++ )
                    {
                        dMTAdmu[nzjis[ o ][ j ]] += this->m_TemplateImage->CalculateRatio(imageValues[o], templateIntensity, positions[o]) * (*imJac) / static_cast< float >( numSamplesOk );
                        ++imJac;
                    }
                }
                
                for( unsigned int o = 0; o < numSamplesOk; o++ )
                {
                    //hier heb ik / static_cast< float >( numSamplesOk ) verwijdert
                    values[positions[o]] += (this->m_TemplateImage->TransformIntensity(imageValues[o], positions[o])-templateIntensity)*(this->m_TemplateImage->TransformIntensity(imageValues[o], positions[o])-templateIntensity);
                    
                    typename DerivativeType::const_iterator imJac = imageJacobians[o].begin();
                    for( unsigned int j = 0; j < nzjis[ o ].size(); ++j )
                    {
                        //hier heb ik / static_cast< float >( numSamplesOk ) verwijdert
                        derivatives[positions[o]][ nzjis[ o ][ j ] ] += ( 2.0 * (this->m_TemplateImage->TransformIntensity(imageValues[o], positions[o]) - templateIntensity ) * (*imJac - dMTAdmu[ nzjis[ o ][ j ] ]));
                        ++imJac;
                    }
                }
            }
        }
        
        this->m_LinearGroupwiseMSDGetValueAndDerivativePerThreadVariables[ threadId ].st_NumberOfPixelsCounted = numberOfPixelsCounted;
        
    }
    
    /**
     * ******************* AfterThreadedGetValueAndDerivative *******************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearGroupwiseMSD< TFixedImage, TMovingImage >
    ::AfterThreadedGetValueAndDerivative(std::vector<MeasureType> & values, std::vector<DerivativeType> & derivatives ) const
    {
        const unsigned int lastDim     = this->GetFixedImage()->GetImageDimension() - 1;
        const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );

        this->m_NumberOfPixelsCounted = this->m_LinearGroupwiseMSDGetValueAndDerivativePerThreadVariables[ 0 ].st_NumberOfPixelsCounted;
        this->m_NumberOfPixelsCountedVector= this->m_LinearGroupwiseMSDGetValueAndDerivativePerThreadVariables[ 0 ].st_NumberOfPixelsCountedVector;
        values = this->m_LinearGroupwiseMSDGetValueAndDerivativePerThreadVariables[ 0 ].st_Values;
        derivatives = this->m_LinearGroupwiseMSDGetValueAndDerivativePerThreadVariables[ 0 ].st_Derivatives;
        
        for( ThreadIdType i = 1; i < this->m_NumberOfThreads; ++i )
        {
            this->m_NumberOfPixelsCounted += this->m_LinearGroupwiseMSDGetValueAndDerivativePerThreadVariables[ i ].st_NumberOfPixelsCounted;
            for(unsigned int d = 0; d < lastDimSize; d++)
            {
                this->m_NumberOfPixelsCountedVector[d] += this->m_LinearGroupwiseMSDGetValueAndDerivativePerThreadVariables[ i ].st_NumberOfPixelsCountedVector[d];
                values[d] += this->m_LinearGroupwiseMSDGetValueAndDerivativePerThreadVariables[ i ].st_Values[d];
                derivatives[d] += this->m_LinearGroupwiseMSDGetValueAndDerivativePerThreadVariables[ i ].st_Derivatives[d];
            }
            
        }
        
        for( unsigned int d = 0; d < lastDimSize; d++ )
        {
            values[d] /= (this->m_InitialVariance * this->m_NumberOfPixelsCountedVector[d]);
            derivatives[d] /= (this->m_InitialVariance * this->m_NumberOfPixelsCountedVector[d]);
        }

        ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
        this->CheckNumberOfSamples( sampleContainer->Size(), this->m_NumberOfPixelsCounted );

    }

}

#endif // end #ifndef _itkLinearGroupwiseMSD_HXX__
