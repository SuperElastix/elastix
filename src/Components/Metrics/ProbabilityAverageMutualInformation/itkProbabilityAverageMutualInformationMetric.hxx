#ifndef _itkProbabilityAverageMutualInformationMetric_HXX__
#define _itkProbabilityAverageMutualInformationMetric_HXX__

#include "itkProbabilityAverageMutualInformationMetric.h"

#ifdef ELASTIX_USE_OPENMP
#include <omp.h>
#endif

namespace itk
{
    /**
     * ********************* Constructor ******************************
     */
    
    template< class TFixedImage, class TMovingImage >
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::ProbabilityAverageMutualInformationMetric()
    {
        
        this->m_ProbabilityAverageMutualInformationMetricThreaderParameters.m_Metric = this;
        
        this->m_ProbabilityAverageMutualInformationMetricGetValueAndDerivativePerThreadVariables     = NULL;
        this->m_ProbabilityAverageMutualInformationMetricGetValueAndDerivativePerThreadVariablesSize = 0;

    }
    
    /**
     * ******************* Destructor *******************
     */
    
    template< class TFixedImage, class TMovingImage >
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::~ProbabilityAverageMutualInformationMetric()
    {
        delete[] this->m_ProbabilityAverageMutualInformationMetricGetValueAndDerivativePerThreadVariables;
    }
    
    /**
     * ******************* Initialize *******************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::Initialize( void ) throw ( ExceptionObject )
    {
        this->Superclass::Initialize();

        this->FillVectors();

        this->InitializeHistograms();

    }

    /**
     * ********************* FillVectors ******************************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::FillVectors(void)
    {
        const unsigned int lastDim     = this->GetFixedImage()->GetImageDimension() - 1;
        const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );

        for(unsigned int d = 0; d < lastDimSize; d++)
        {
            MovingImagePixelType trueMinTemp = NumericTraits< MovingImagePixelType >::max();
            MovingImagePixelType trueMaxTemp = NumericTraits< MovingImagePixelType >::NonpositiveMin();
            
            /** If no mask. */
            if( this->m_MovingImageMask.IsNull() )
            {
                typedef ImageRegionConstIterator< MovingImageType > IteratorType;
                
                MovingImageRegionType region = this->GetMovingImage()->GetLargestPossibleRegion();
                region.SetIndex(ReducedMovingImageDimension, d);
                region.SetSize(ReducedMovingImageDimension, 1);
                IteratorType it(this->GetMovingImage(), region );
                
                for( it.GoToBegin(); !it.IsAtEnd(); ++it )
                {
                    const MovingImagePixelType sample = it.Get();
                    trueMinTemp = vnl_math_min( trueMinTemp, sample );
                    trueMaxTemp = vnl_math_max( trueMaxTemp, sample );
                    
                }
            }
            
            else
            {
                typedef ImageRegionConstIteratorWithIndex< MovingImageType > IteratorType;
                
                MovingImageRegionType region = this->GetMovingImage()->GetLargestPossibleRegion();
                
                region.SetIndex(ReducedMovingImageDimension, d);
                region.SetSize(ReducedMovingImageDimension, 1);
                
                IteratorType it(this->GetMovingImage(), region );
                
                for( it.GoToBegin(); !it.IsAtEnd(); ++it )
                {
                    OutputPointType point;
                    this->GetMovingImage()->TransformIndexToPhysicalPoint( it.GetIndex(), point );
                    if( this->IsInsideMovingMask( point ) )
                    {
                        const MovingImagePixelType sample = it.Get();
                        trueMinTemp = vnl_math_min( trueMinTemp, sample );
                        trueMaxTemp = vnl_math_max( trueMaxTemp, sample );
                        
                    }
                    
                }
            }
            
            this->m_FixedImageTrueMins[d] = trueMinTemp;
            this->m_FixedImageTrueMaxs[d] = trueMaxTemp;
            
            this->m_FixedImageMinLimits[d] = static_cast< MovingImageLimiterOutputType >(trueMinTemp - this->m_FixedImageLimitRangeRatio * ( trueMaxTemp - trueMinTemp ) );
            this->m_FixedImageMaxLimits[d] =  static_cast< MovingImageLimiterOutputType >(trueMaxTemp + this->m_FixedImageLimitRangeRatio * ( trueMaxTemp - trueMinTemp ) );

        }
        
        for(unsigned int d = 0; d < lastDimSize; d++)
        {
            MovingImagePixelType trueMinTemp = NumericTraits< MovingImagePixelType >::max();
            MovingImagePixelType trueMaxTemp = NumericTraits< MovingImagePixelType >::NonpositiveMin();
            
            for(unsigned int i = 0; i < lastDimSize; i++)
            {
                if ( i != d )
                {
                    trueMinTemp = vnl_math_min( trueMinTemp, this->m_FixedImageTrueMins[i] );
                    trueMaxTemp = vnl_math_max( trueMaxTemp, this->m_FixedImageTrueMaxs[i] );
                }
            }
            
            
            this->m_MovingImageTrueMins[d] = trueMinTemp;
            this->m_MovingImageTrueMaxs[d] = trueMaxTemp;

            this->m_MovingImageMinLimits[d] = static_cast< MovingImageLimiterOutputType >( this->m_MovingImageTrueMins[d] - this->m_MovingImageLimitRangeRatio * ( this->m_MovingImageTrueMaxs[d] - this->m_MovingImageTrueMins[d] ));
            this->m_MovingImageMaxLimits[d] = static_cast< MovingImageLimiterOutputType >( this->m_MovingImageTrueMaxs[d] + this->m_MovingImageLimitRangeRatio * ( this->m_MovingImageTrueMaxs[d] - this->m_MovingImageTrueMins[d] ));

        }
        
        
        for(unsigned int d = 0; d < lastDimSize; d++)
        {
            this->m_FixedImageMarginalPDFs[d] = new MarginalPDFType;
            this->m_FixedImageMarginalPDFs[d]->SetSize(this->m_NumberOfFixedHistogramBins);
            this->m_MovingImageMarginalPDFs[d] = new MarginalPDFType;
            this->m_MovingImageMarginalPDFs[d]->SetSize( this->m_NumberOfMovingHistogramBins );
            this->m_JointPDFs[d] = JointPDFType::New();
        }
        
    }
    
    /**
     * ********************* InitializeHistograms ******************************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::InitializeHistograms(void)
    {
        const unsigned int lastDim     = this->GetFixedImage()->GetImageDimension() - 1;
        const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );

        int fixedPadding  = this->m_FixedKernelBSplineOrder / 2;
        int movingPadding = this->m_MovingKernelBSplineOrder / 2;
        
        const double smallNumberRatio = 0.001;
        
        for(unsigned int d = 0; d < lastDimSize; d++)
        {
            const double fixedHistogramWidth = static_cast< double >( static_cast< OffsetValueType >( this->m_NumberOfFixedHistogramBins ) - 2.0 * fixedPadding - 1.0 );
            const double movingHistogramWidth = static_cast< double >( static_cast< OffsetValueType >( this->m_NumberOfMovingHistogramBins ) - 2.0 * movingPadding - 1.0 );

            double smallNumberMoving = smallNumberRatio * ( this->m_MovingImageMaxLimits[d] - this->m_MovingImageMinLimits[d] ) / static_cast< double >( this->m_NumberOfMovingHistogramBins - 2 * movingPadding - 1 );
            
            this->m_MovingImageBinSizes[d] = ( this->m_MovingImageMaxLimits[d] - this->m_MovingImageMinLimits[d] + 2.0 * smallNumberMoving ) / movingHistogramWidth;
            
            this->m_MovingImageBinSizes[d] = vnl_math_max( this->m_MovingImageBinSizes[d], 1e-10 );
            this->m_MovingImageBinSizes[d] = vnl_math_min( this->m_MovingImageBinSizes[d], 1e+10 );
            this->m_MovingImageNormalizedMins[d] = ( this->m_MovingImageMinLimits[d] - smallNumberMoving ) / this->m_MovingImageBinSizes[d] - static_cast< double >( movingPadding );
            
            const double smallNumberFixed = smallNumberRatio * ( this->m_FixedImageMaxLimits[d] - this->m_FixedImageMinLimits[d] ) / static_cast< double >( this->m_NumberOfFixedHistogramBins - 2 * fixedPadding - 1 );
            this->m_FixedImageBinSizes[d] = static_cast<double>( this->m_FixedImageMaxLimits[d] - this->m_FixedImageMinLimits[d]+ 2.0 * smallNumberFixed ) / fixedHistogramWidth;
            
            this->m_FixedImageBinSizes[d] = vnl_math_max( this->m_FixedImageBinSizes[d], 1e-10 );
            this->m_FixedImageBinSizes[d] = vnl_math_min( this->m_FixedImageBinSizes[d], 1e+10 );
            this->m_FixedImageNormalizedMins[d] = (this->m_FixedImageMinLimits[d] - smallNumberFixed ) / this->m_FixedImageBinSizes[d] - static_cast< double >( fixedPadding );
            
        }

        
        JointPDFRegionType jointPDFRegion;
        JointPDFIndexType  jointPDFIndex;
        JointPDFSizeType   jointPDFSize;
        jointPDFIndex.Fill( 0 );
        jointPDFSize[ 0 ] = this->m_NumberOfMovingHistogramBins;
        jointPDFSize[ 1 ] = this->m_NumberOfFixedHistogramBins;
        jointPDFRegion.SetIndex( jointPDFIndex );
        jointPDFRegion.SetSize( jointPDFSize );
        
        for(unsigned int d = 0; d < lastDimSize; d++)
        {
            this->m_JointPDFs[d]->SetRegions( jointPDFRegion );
            this->m_JointPDFs[d]->Allocate();
            this->m_JointPDFs[d]->FillBuffer(0.0);
            
            if( this->GetUseDerivative() )
            {
                JointPDFDerivativesRegionType jointPDFDerivativesRegion;
                JointPDFDerivativesIndexType  jointPDFDerivativesIndex;
                JointPDFDerivativesSizeType   jointPDFDerivativesSize;
                jointPDFDerivativesIndex.Fill( 0 );
                jointPDFDerivativesSize[ 0 ] = this->GetNumberOfParameters();
                jointPDFDerivativesSize[ 1 ] = this->m_NumberOfMovingHistogramBins;
                jointPDFDerivativesSize[ 2 ] = this->m_NumberOfFixedHistogramBins;
                jointPDFDerivativesRegion.SetIndex( jointPDFDerivativesIndex );
                jointPDFDerivativesRegion.SetSize( jointPDFDerivativesSize );
                
                if( this->m_UseExplicitPDFDerivatives )
                {
                    this->m_JointPDFDerivatives[d] = JointPDFDerivativesType::New();
                    this->m_JointPDFDerivatives[d]->SetRegions( jointPDFDerivativesRegion );
                    this->m_JointPDFDerivatives[d]->Allocate();
                    this->m_JointPDFDerivatives[d]->FillBuffer(0.0);
                }
                
                else
                {
                    if( !this->m_JointPDFDerivatives[d].IsNull() )
                    {
                        jointPDFDerivativesSize.Fill( 0 );
                        jointPDFDerivativesRegion.SetSize( jointPDFDerivativesSize );
                        this->m_JointPDFDerivatives[d]->SetRegions( jointPDFDerivativesRegion );
                        this->m_JointPDFDerivatives[d]->Allocate();
                        this->m_JointPDFDerivatives[d]->GetPixelContainer()->Squeeze();
                    }
                }
                
            }
            
            else
            {
                this->m_JointPDFDerivatives[d]  = 0;
            }	
            
        }
        
        if( !this->GetUseExplicitPDFDerivatives() )
        {
            this->m_PRatioArray.resize(lastDimSize);
            
            for(unsigned int d = 0; d < lastDimSize; d++)
            {
                this->m_PRatioArray[d] = new PRatioArrayType;
                this->m_PRatioArray[d]->SetSize( this->GetNumberOfFixedHistogramBins(), this->GetNumberOfMovingHistogramBins() );
                this->m_PRatioArray[d]->Fill( itk::NumericTraits< PRatioType >::Zero );
                
            }
        }
    }
    
    
    /**
     * ********************* GetValue ****************************
     */
    
    template< class TFixedImage, class TMovingImage >
    typename PAverageMutualInformationMetric< TFixedImage, TMovingImage >::MeasureType
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::GetValue( const ParametersType & parameters ) const
    {
        this->ComputePDFs( parameters );
        
        for(unsigned int s = 0; s < this->m_NumSamplesLastDimension; s++)
        {
            this->NormalizeJointPDF( this->m_JointPDFs[this->m_RandomList[s]], 1.0/static_cast<double>(this->m_NumberOfPixelsCountedVector[this->m_RandomList[s]]) );
            this->ComputeMarginalPDF( this->m_JointPDFs[this->m_RandomList[s]], this->m_FixedImageMarginalPDFs[this->m_RandomList[s]], 0 );
            this->ComputeMarginalPDF( this->m_JointPDFs[this->m_RandomList[s]], this->m_MovingImageMarginalPDFs[this->m_RandomList[s]], 1 );
        }
        
        typedef ImageLinearConstIteratorWithIndex< JointPDFType >           JointPDFIteratorType;
        typedef ImageLinearConstIteratorWithIndex<JointPDFDerivativesType >JointPDFDerivativesIteratorType;
        typedef typename MarginalPDFType::const_iterator                    MarginalPDFIteratorType;
        
        double PAMI =0.0;
        for(unsigned int s = 0; s < this->m_NumSamplesLastDimension; s++)
        {
            JointPDFIteratorType jointPDFit(this->m_JointPDFs[this->m_RandomList[s]], this->m_JointPDFs[this->m_RandomList[s]]->GetLargestPossibleRegion() );
            jointPDFit.SetDirection( 0 );
            jointPDFit.GoToBegin();
            
            MarginalPDFIteratorType       fixedPDFit   = this->m_FixedImageMarginalPDFs[this->m_RandomList[s]]->begin();
            const MarginalPDFIteratorType fixedPDFend  = this->m_FixedImageMarginalPDFs[this->m_RandomList[s]]->end();
            MarginalPDFIteratorType       movingPDFit  = this->m_MovingImageMarginalPDFs[this->m_RandomList[s]]->begin();
            const MarginalPDFIteratorType movingPDFend = this->m_MovingImageMarginalPDFs[this->m_RandomList[s]]->end();
            
            double MI = 0.0;
            
            while( fixedPDFit != fixedPDFend )
            {
                const double fixedImagePDFValue = *fixedPDFit;
                movingPDFit = this->m_MovingImageMarginalPDFs[this->m_RandomList[s]]->begin();
                
                while( movingPDFit != movingPDFend )
                {
                    const double movingImagePDFValue = *movingPDFit;
                    const double fixPDFmovPDF        = fixedImagePDFValue * movingImagePDFValue;
                    const double jointPDFValue       = jointPDFit.Get();
                    
                    if( jointPDFValue > 1e-16 && fixPDFmovPDF > 1e-16 )
                    {
                        const double pRatio      = vcl_log( jointPDFValue / fixPDFmovPDF );
                        const double pRatioAlpha = pRatio /static_cast<double>(this->m_NumberOfPixelsCountedVector[this->m_RandomList[s]]);
                        MI = MI + jointPDFValue * pRatio;
                    }
                    
                    ++movingPDFit;
                    ++jointPDFit;
                    
                }
                ++fixedPDFit;
                jointPDFit.NextLine();
                
            }
            PAMI += MI;
        }
        
        return static_cast< MeasureType >( -1.0 * PAMI / static_cast<float>(this->m_NumSamplesLastDimension) );

    }
    
    /**
     * ********************* GetDerivative ****************************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::GetDerivative(const ParametersType & parameters, DerivativeType & derivative ) const
    {
        MeasureType value;
        this->GetValueAndDerivative( parameters, value, derivative );
    }
    
    
    /**
     * *********************  GetValueAndDerivative ****************************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::GetValueAndDerivative( const ParametersType & parameters, MeasureType & value, DerivativeType & derivative ) const
    {
        if ( this->GetUseExplicitPDFDerivatives() )
        {
            this->GetValueAndAnalyticalDerivative(parameters, value, derivative);
        }
        else
        {
            this->GetValueAndAnalyticalDerivativeLowMemory(parameters, value, derivative );
        }
    }
    
    /**
     * *********************  GetValueAndAnalyticalDerivative ****************************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::GetValueAndAnalyticalDerivative( const ParametersType & parameters, MeasureType & value, DerivativeType & derivative ) const
    {
        value      = NumericTraits< MeasureType >::Zero;
        derivative = DerivativeType( this->GetNumberOfParameters() );
        derivative.Fill( NumericTraits< double >::Zero );
		      
        this->ComputePDFsAndPDFDerivatives( parameters);
        
        for(unsigned int s = 0; s < this->m_NumSamplesLastDimension; s++)
        {
            this->NormalizeJointPDF( this->m_JointPDFs[this->m_RandomList[s]], 1.0/static_cast<double>(this->m_NumberOfPixelsCountedVector[this->m_RandomList[s]]) );
            this->ComputeMarginalPDF( this->m_JointPDFs[this->m_RandomList[s]], this->m_FixedImageMarginalPDFs[this->m_RandomList[s]], 0 );
            this->ComputeMarginalPDF( this->m_JointPDFs[this->m_RandomList[s]], this->m_MovingImageMarginalPDFs[this->m_RandomList[s]], 1 );
        }
        
        typedef ImageLinearConstIteratorWithIndex< JointPDFType > JointPDFIteratorType;
        typedef ImageLinearConstIteratorWithIndex<JointPDFDerivativesType >                       JointPDFDerivativesIteratorType;
        typedef typename MarginalPDFType::const_iterator          MarginalPDFIteratorType;
        
        typedef typename DerivativeType::iterator        DerivativeIteratorType;
        typedef typename DerivativeType::const_iterator  DerivativeConstIteratorType;
        
        derivative.Fill(0.0);
        double PAMI = 0.0;
        
        for(unsigned int s = 0; s < this->m_NumSamplesLastDimension; s++)
        {
            JointPDFIteratorType jointPDFit(this->m_JointPDFs[this->m_RandomList[s]], this->m_JointPDFs[this->m_RandomList[s]]->GetLargestPossibleRegion() );
            jointPDFit.SetDirection( 0 );
            jointPDFit.GoToBegin();
            JointPDFDerivativesIteratorType jointPDFDerivativesit(this->m_JointPDFDerivatives[this->m_RandomList[s]], this->m_JointPDFDerivatives[this->m_RandomList[s]]->GetLargestPossibleRegion());
            jointPDFDerivativesit.SetDirection( 0 );
            jointPDFDerivativesit.GoToBegin();
            MarginalPDFIteratorType       fixedPDFit   = this->m_FixedImageMarginalPDFs[this->m_RandomList[s]]->begin();
            const MarginalPDFIteratorType fixedPDFend  = this->m_FixedImageMarginalPDFs[this->m_RandomList[s]]->end();
            MarginalPDFIteratorType       movingPDFit  = this->m_MovingImageMarginalPDFs[this->m_RandomList[s]]->begin();
            const MarginalPDFIteratorType movingPDFend = this->m_MovingImageMarginalPDFs[this->m_RandomList[s]]->end();
            
            DerivativeIteratorType        derivit      = derivative.begin();
            const DerivativeIteratorType  derivbegin   = derivative.begin();
            const DerivativeIteratorType  derivend     = derivative.end();
            
            double MI = 0.0;
            
            while( fixedPDFit != fixedPDFend )
            {
                const double fixedImagePDFValue = *fixedPDFit;
                movingPDFit = this->m_MovingImageMarginalPDFs[this->m_RandomList[s]]->begin();
                while( movingPDFit != movingPDFend )
                {
                    const double movingImagePDFValue = *movingPDFit;
                    const double fixPDFmovPDF        = fixedImagePDFValue * movingImagePDFValue;
                    const double jointPDFValue       = jointPDFit.Get();
                    
                    if( jointPDFValue > 1e-16 && fixPDFmovPDF > 1e-16 )
                    {
                        derivit = derivbegin;
                        const double pRatio      = vcl_log( jointPDFValue / fixPDFmovPDF );
                        const double pRatioAlpha = pRatio / static_cast<double>(this->m_NumberOfPixelsCountedVector[this->m_RandomList[s]]);
                        MI = MI + jointPDFValue * pRatio;
                        while( derivit != derivend )
                        {
                            ( *derivit ) -= jointPDFDerivativesit.Get() * pRatioAlpha / static_cast<float>(this->m_NumSamplesLastDimension);
                            ++derivit;
                            ++jointPDFDerivativesit;
                        }
                        
                    }
                    
                    ++movingPDFit;
                    ++jointPDFit;
                    jointPDFDerivativesit.NextLine();
                    
                }
                
                ++fixedPDFit;
                jointPDFit.NextLine();
                
            }
            PAMI += MI;
            
        }
        

        if( this->m_SubtractMean )
        {
            if( !this->m_TransformIsStackTransform )
            {
                const unsigned int lastDimGridSize              = this->m_GridSize[ this->ReducedFixedImageDimension ];
                const unsigned int numParametersPerDimension    = this->GetNumberOfParameters() / this->GetMovingImage()->GetImageDimension();
                const unsigned int numControlPointsPerDimension = numParametersPerDimension / lastDimGridSize;
                DerivativeType     mean( numControlPointsPerDimension );
                for( unsigned int d = 0; d < this->GetMovingImage()->GetImageDimension(); ++d )
                {
                    mean.Fill( 0.0 );
                    const unsigned int starti = numParametersPerDimension * d;
                    for( unsigned int i = starti; i < starti + numParametersPerDimension; ++i )
                    {
                        const unsigned int index = i % numControlPointsPerDimension;
                        mean[ index ] += derivative[ i ];
                    }
                    mean /= static_cast< double >( lastDimGridSize );
                    
                    for( unsigned int i = starti; i < starti + numParametersPerDimension; ++i )
                    {
                        const unsigned int index = i % numControlPointsPerDimension;
                        derivative[ i ] -= mean[ index ];
                    }
                }
            }
            else
            {
                const unsigned int numParametersPerLastDimension = this->GetNumberOfParameters() / this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension);
                DerivativeType     mean( numParametersPerLastDimension );
                mean.Fill( 0.0 );
                
                for( unsigned int t = 0; t < this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension); ++t )
                {
                    const unsigned int startc = numParametersPerLastDimension * t;
                    for( unsigned int c = startc; c < startc + numParametersPerLastDimension; ++c )
                    {
                        const unsigned int index = c % numParametersPerLastDimension;
                        mean[ index ] += derivative[ c ];
                    }
                }
                mean /= static_cast< double >( this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension) );
                
                for( unsigned int t = 0; t < this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension); ++t )
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

        value = static_cast< MeasureType >( -1.0 * PAMI / static_cast<float>(this->m_NumSamplesLastDimension));
        

    }
    
    /**
     * ********************  GetValueAndAnalyticalDerivativeLowMemory **************************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::GetValueAndAnalyticalDerivativeLowMemory( const ParametersType & parameters, MeasureType & value, DerivativeType & derivative ) const
    {
        
        value = NumericTraits< MeasureType >::Zero;
        derivative = DerivativeType( this->GetNumberOfParameters() );
        derivative.Fill( NumericTraits<double>::Zero );
        
        this->ComputePDFs( parameters );
        
        double MI = 0.0;
        double PAMI = 0.0;
        
        for(unsigned int s = 0; s < this->m_NumSamplesLastDimension; s++)
        {
            this->NormalizeJointPDF( this->m_JointPDFs[this->m_RandomList[s]], 1.0/static_cast<double>(this->m_NumberOfPixelsCountedVector[this->m_RandomList[s]]) );
            this->ComputeMarginalPDF( this->m_JointPDFs[this->m_RandomList[s]], this->m_FixedImageMarginalPDFs[this->m_RandomList[s]], 0 );
            this->ComputeMarginalPDF( this->m_JointPDFs[this->m_RandomList[s]], this->m_MovingImageMarginalPDFs[this->m_RandomList[s]], 1 );
            this->ComputeValueAndPRatioArray( MI, s);
            PAMI += MI / static_cast<float>(this->m_NumSamplesLastDimension);

        }
        
        this->ComputeDerivativeLowMemory( derivative);
        
        if( this->m_SubtractMean )
        {
            if( !this->m_TransformIsStackTransform )
            {
                const unsigned int lastDimGridSize              = this->m_GridSize[ this->ReducedFixedImageDimension ];
                const unsigned int numParametersPerDimension    = this->GetNumberOfParameters() / this->GetMovingImage()->GetImageDimension();
                const unsigned int numControlPointsPerDimension = numParametersPerDimension / lastDimGridSize;
                DerivativeType     mean( numControlPointsPerDimension );
                for( unsigned int d = 0; d < this->GetMovingImage()->GetImageDimension(); ++d )
                {
                    mean.Fill( 0.0 );
                    const unsigned int starti = numParametersPerDimension * d;
                    for( unsigned int i = starti; i < starti + numParametersPerDimension; ++i )
                    {
                        const unsigned int index = i % numControlPointsPerDimension;
                        mean[ index ] += derivative[ i ];
                    }
                    mean /= static_cast< double >( lastDimGridSize );
                    
                    for( unsigned int i = starti; i < starti + numParametersPerDimension; ++i )
                    {
                        const unsigned int index = i % numControlPointsPerDimension;
                        derivative[ i ] -= mean[ index ];
                    }
                    
                }
                
            }
            else
            {
                const unsigned int numParametersPerLastDimension = this->GetNumberOfParameters() / this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension);
                DerivativeType     mean( numParametersPerLastDimension );
                mean.Fill( 0.0 );
                
                for( unsigned int t = 0; t < this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension); ++t )
                {
                    const unsigned int startc = numParametersPerLastDimension * t;
                    for( unsigned int c = startc; c < startc + numParametersPerLastDimension; ++c )
                    {
                        const unsigned int index = c % numParametersPerLastDimension;
                        mean[ index ] += derivative[ c ];
                    }
                }
                mean /= static_cast< double >( this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension) );
                
                for( unsigned int t = 0; t < this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension); ++t )
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

        value = static_cast< MeasureType >( -1.0 * PAMI );


    }
    
    
    /**
     * ************************ ComputePDFs **************************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::ComputePDFs( const ParametersType & parameters ) const
    {
        if( !this->m_UseMultiThread )
        {
            return this->ComputePDFsSingleThreaded( parameters );
        }
        
        this->BeforeThreadedGetValueAndDerivative( parameters );
        
        this->InitializeThreadingParameters();
        
        this->LaunchComputePDFsThreaderCallback();
        
        this->AfterThreadedComputePDFs();
        
    }
    
    /**
     * **************** ComputePDFsThreaderCallback *******
     */
    
    template< class TFixedImage, class TMovingImage >
    ITK_THREAD_RETURN_TYPE
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::ComputePDFsThreaderCallback( void * arg )
    {
        ThreadInfoType * infoStruct = static_cast< ThreadInfoType * >( arg );
        ThreadIdType     threadId   = infoStruct->ThreadID;
        
        ProbabilityAverageMutualInformationMetricMultiThreaderParameterType * temp
        = static_cast< ProbabilityAverageMutualInformationMetricMultiThreaderParameterType * >( infoStruct->UserData );
        
        temp->m_Metric->ThreadedComputePDFs( threadId );
        
        return ITK_THREAD_RETURN_VALUE;

    }
    
    /**
     * *********************** LaunchComputePDFsThreaderCallback***************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::LaunchComputePDFsThreaderCallback( void ) const
    {
        typename ThreaderType::Pointer local_threader = ThreaderType::New();
        local_threader->SetNumberOfThreads( this->m_NumberOfThreads );
        local_threader->SetSingleMethod( this->ComputePDFsThreaderCallback, const_cast< void * >( static_cast< const void * >( &this->m_ProbabilityAverageMutualInformationMetricThreaderParameters ) ) );
        
        local_threader->SingleMethodExecute();

    }
    
    /**
     * ********************* InitializeThreadingParameters ****************************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::InitializeThreadingParameters( void ) const
    {
        const unsigned int lastDim     = this->GetFixedImage()->GetImageDimension() - 1;
        const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );
        
        Superclass::InitializeThreadingParameters();
        
        JointPDFRegionType jointPDFRegion;
        JointPDFIndexType  jointPDFIndex;
        JointPDFSizeType   jointPDFSize;
        jointPDFIndex.Fill( 0 );
        jointPDFSize[ 0 ] = this->m_NumberOfMovingHistogramBins;
        jointPDFSize[ 1 ] = this->m_NumberOfFixedHistogramBins;
        jointPDFRegion.SetIndex( jointPDFIndex );
        jointPDFRegion.SetSize( jointPDFSize );
        
        if( this->m_ProbabilityAverageMutualInformationMetricGetValueAndDerivativePerThreadVariablesSize != this->m_NumberOfThreads )
        {
            delete[] this->m_ProbabilityAverageMutualInformationMetricGetValueAndDerivativePerThreadVariables;
            this->m_ProbabilityAverageMutualInformationMetricGetValueAndDerivativePerThreadVariables = new AlignedProbabilityAverageMutualInformationMetricGetValueAndDerivativePerThreadStruct[this->m_NumberOfThreads ];
            this->m_ProbabilityAverageMutualInformationMetricGetValueAndDerivativePerThreadVariablesSize = this->m_NumberOfThreads;
        }
        
        for( ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i )
        {
            this->m_ProbabilityAverageMutualInformationMetricGetValueAndDerivativePerThreadVariables[ i ].st_NumberOfPixelsCounted = NumericTraits< SizeValueType >::Zero;
            this->m_ProbabilityAverageMutualInformationMetricGetValueAndDerivativePerThreadVariables[ i ].st_NumberOfPixelsCountedVector.assign(this->m_NumSamplesLastDimension, NumericTraits< SizeValueType >::Zero);
            this->m_ProbabilityAverageMutualInformationMetricGetValueAndDerivativePerThreadVariables[ i ].st_JointPDFs.resize(lastDimSize);
            
            for(unsigned int d = 0; d < lastDimSize; d++)
            {
                JointPDFPointer & jointPDF = this->m_ProbabilityAverageMutualInformationMetricGetValueAndDerivativePerThreadVariables[ i ].st_JointPDFs[ d ];
                if( jointPDF.IsNull() ) { jointPDF = JointPDFType::New(); }
                if( jointPDF->GetLargestPossibleRegion() != jointPDFRegion )
                {
                    jointPDF->SetRegions( jointPDFRegion );
                    jointPDF->Allocate();
                }
                
            }
            
            
        }

    } // end InitializeThreadingParameters()

    
    /**
     * ******************* AfterThreadedComputePDFs *******************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::AfterThreadedComputePDFs( void ) const
    {
        this->m_NumberOfPixelsCountedVector.assign(this->m_NumSamplesLastDimension,0.0);
        for( ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i )
        {
            this->m_NumberOfPixelsCounted += this->m_ProbabilityAverageMutualInformationMetricGetValueAndDerivativePerThreadVariables[ i ].st_NumberOfPixelsCounted;
            
            for(unsigned int s = 0; s < this->m_NumSamplesLastDimension; s++)
            {
                this->m_NumberOfPixelsCountedVector[this->m_RandomList[s]] += this->m_ProbabilityAverageMutualInformationMetricGetValueAndDerivativePerThreadVariables[ i ].st_NumberOfPixelsCountedVector[this->m_RandomList[s]];
            }
            
        }
        
        ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
        this->CheckNumberOfSamples( sampleContainer->Size(), this->m_NumberOfPixelsCounted );
        
        for(unsigned int s = 0; s < this->m_NumSamplesLastDimension; s++)
        {
            typedef ImageScanlineIterator< JointPDFType > JointPDFIteratorType;
            JointPDFIteratorType                it( this->m_JointPDFs[this->m_RandomList[s]], this->m_JointPDFs[this->m_RandomList[s]]->GetBufferedRegion() );
            std::vector< JointPDFIteratorType > itT( this->m_NumberOfThreads );
            for( ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i )
            {
                itT[ i ] = JointPDFIteratorType(this-> m_ProbabilityAverageMutualInformationMetricGetValueAndDerivativePerThreadVariables[ i ].st_JointPDFs[this->m_RandomList[s]], this->m_JointPDFs[this->m_RandomList[s]]->GetBufferedRegion() );
            }
            
            PDFValueType sum;
            while( !it.IsAtEnd() )
            {
                while( !it.IsAtEndOfLine() )
                {
                    sum = NumericTraits< PDFValueType >::Zero;
                    for( ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i )
                    {
                        sum += itT[ i ].Value();
                        ++itT[ i ];
                    }
                    it.Set( sum );
                    ++it;
                }
                it.NextLine();
                for( ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i )
                {
                    itT[ i ].NextLine();
                }
            }
            
        }

    }

    
    /**
     * ************************ ComputePDFsSingleThreaded **************************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::ComputePDFsSingleThreaded( const ParametersType & parameters ) const
    {
        const unsigned int lastDim     = this->GetFixedImage()->GetImageDimension() - 1;
        const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );
        
        for(unsigned int s = 0; s < this->m_NumSamplesLastDimension; s++)
        {
            this->m_JointPDFs[this->m_RandomList[s]]->FillBuffer( 0.0 );
        }
        
        this->m_NumberOfPixelsCountedVector.assign(this->m_NumSamplesLastDimension,0.0);
        this->m_NumberOfPixelsCounted = 0;
        
        this->BeforeThreadedGetValueAndDerivative( parameters );
        
        ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
        
        typename ImageSampleContainerType::ConstIterator fiter;
        typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
        typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();
        
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
                RealType             movingImageValueTemp;
                MovingImagePointType mappedPoint;
                
                voxelCoord[ lastDim ] = this->m_RandomList[ s ];
                
                this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord, fixedPoint );
                
                bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );
                
                if( sampleOk )
                {
                    sampleOk = this->IsInsideMovingMask( mappedPoint );
                }
                
                if( sampleOk )
                {
                    sampleOk = this->EvaluateMovingImageValueAndDerivative(mappedPoint, movingImageValueTemp, 0 );
                }
                
                if( sampleOk )
                {
                    if (movingImageValueTemp > this->m_FixedImageTrueMaxs[this->m_RandomList[ s ]])
                    {
                        imageValues[numSamplesOk] = this->m_FixedImageTrueMaxs[this->m_RandomList[ s ]];
                    }
                    else if (movingImageValueTemp < this->m_FixedImageTrueMins[this->m_RandomList[ s ]] )
                    {
                        imageValues[numSamplesOk] = this->m_FixedImageTrueMins[this->m_RandomList[ s ]];
                    }
                    else
                    {
                        imageValues[numSamplesOk] = movingImageValueTemp;
                    }

                    positions[numSamplesOk] = this->m_RandomList[ s ];
                    numSamplesOk++;
                }
                
            }
            
            if ( numSamplesOk > 1)
            {
                this->m_NumberOfPixelsCounted++;
                imageValues.resize(numSamplesOk);
                positions.resize(numSamplesOk);
                
                for( unsigned int o = 0; o < numSamplesOk; o++ )
                {
                    this->m_NumberOfPixelsCountedVector[positions[o]]++;
                }

                this->UpdateJointPDFAndDerivatives(imageValues, &this->m_JointPDFs, 0, 0, numSamplesOk, positions);

            }

        }
        
        this->CheckNumberOfSamples( sampleContainer->Size(), this->m_NumberOfPixelsCounted );

        
    } // end ComputePDFs
    
    /**
     * ******************* ThreadedComputePDFs *******************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::ThreadedComputePDFs( ThreadIdType threadId )
    {
        const unsigned int lastDim     = this->GetFixedImage()->GetImageDimension() - 1;
        const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );
        
        std::vector<JointPDFPointer> & jointPDFs = this->m_ProbabilityAverageMutualInformationMetricGetValueAndDerivativePerThreadVariables[ threadId ].st_JointPDFs;
        std::vector<SizeValueType> & NumberOfPixelsCountedVector = this->m_ProbabilityAverageMutualInformationMetricGetValueAndDerivativePerThreadVariables[ threadId ].st_NumberOfPixelsCountedVector;
        SizeValueType & NumberOfPixelsCounted = this->m_ProbabilityAverageMutualInformationMetricGetValueAndDerivativePerThreadVariables[ threadId ].st_NumberOfPixelsCounted;
        
        for(unsigned int s = 0; s < this->m_NumSamplesLastDimension; s++)
        {
            jointPDFs[this->m_RandomList[s]]->FillBuffer( 0.0 );
        }
        
        NumberOfPixelsCountedVector.assign(this->m_NumSamplesLastDimension,0.0);
        NumberOfPixelsCounted = 0;
        
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
            std::vector< double >  imageValues(this->m_NumSamplesLastDimension);
            std::vector< unsigned int > positions(this->m_NumSamplesLastDimension);
            
            for( unsigned int s = 0; s < this->m_NumSamplesLastDimension; s++ )
            {
                RealType             movingImageValueTemp;
                MovingImagePointType mappedPoint;
                
                voxelCoord[ lastDim ] = this->m_RandomList[ s ];
                
                this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord, fixedPoint );
                
                bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );
                
                if( sampleOk )
                {
                    sampleOk = this->IsInsideMovingMask( mappedPoint );
                }
                
                if( sampleOk )
                {
                    sampleOk = this->EvaluateMovingImageValueAndDerivative(mappedPoint, movingImageValueTemp, 0 );
                }
                
                if( sampleOk )
                {
                    if (movingImageValueTemp > this->m_FixedImageTrueMaxs[this->m_RandomList[ s ]])
                    {
                        imageValues[numSamplesOk] = this->m_FixedImageTrueMaxs[this->m_RandomList[ s ]];
                    }
                    else if (movingImageValueTemp < this->m_FixedImageTrueMins[this->m_RandomList[ s ]] )
                    {
                        imageValues[numSamplesOk] = this->m_FixedImageTrueMins[this->m_RandomList[ s ]];
                    }
                    else
                    {
                        imageValues[numSamplesOk] = movingImageValueTemp;
                    }
                    positions[numSamplesOk] = this->m_RandomList[ s ];
                    numSamplesOk++;
                }
                
            }
            
            if ( numSamplesOk > 1)
            {
                NumberOfPixelsCounted++;
                imageValues.resize(numSamplesOk);
                positions.resize(numSamplesOk);
                
                for( unsigned int o = 0; o < numSamplesOk; o++ )
                {
                    NumberOfPixelsCountedVector[positions[o]]++;
                }
                
                this->UpdateJointPDFAndDerivatives(imageValues, &jointPDFs, 0, 0, numSamplesOk, positions);
                
            }
            
        }

    }
    
    /**
     * ************************ ComputePDFsAndPDFDerivatives **************************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::ComputePDFsAndPDFDerivatives( const ParametersType & parameters ) const
    {
        //if( !this->m_UseMultiThread )
        //{
            return this->ComputePDFsAndPDFDerivativesSingleThreaded( parameters );
        //}

        //this->BeforeThreadedGetValueAndDerivative( parameters );
        
        //this->InitializeThreadingParameters();
        
        //this->LaunchComputePDFsAndPDFDerivativesThreaderCallback();
        
        //this->AfterThreadedComputePDFsAndPDFDerivatives();
        
    }
    
    /**
     * ************************ ComputePDFsAndPDFDerivativesSingleThreaded *******************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::ComputePDFsAndPDFDerivativesSingleThreaded( const ParametersType & parameters ) const
    {
        const unsigned int lastDim     = this->GetFixedImage()->GetImageDimension() - 1;
        const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );
        
        for(unsigned int s = 0; s < this->m_NumSamplesLastDimension; s++)
        {
            this->m_JointPDFs[this->m_RandomList[s]]->FillBuffer( 0.0 );
            this->m_JointPDFDerivatives[this->m_RandomList[s]]->FillBuffer(0.0);
        }
        
        typedef typename DerivativeType::ValueType        DerivativeValueType;
        typedef typename TransformJacobianType::ValueType TransformJacobianValueType;

        this->m_NumberOfPixelsCountedVector.assign(this->m_NumSamplesLastDimension,0.0);
        this->m_NumberOfPixelsCounted = 0;
        
        this->BeforeThreadedGetValueAndDerivative( parameters );
        
        ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
        
        typename ImageSampleContainerType::ConstIterator fiter;
        typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
        typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();
        
        for( fiter = fbegin; fiter != fend; ++fiter )
        {
            FixedImagePointType fixedPoint = ( *fiter ).Value().m_ImageCoordinates;
            FixedImageContinuousIndexType       voxelCoord;
            this->GetFixedImage()->TransformPhysicalPointToContinuousIndex( fixedPoint, voxelCoord );
            
            TransformJacobianType  jacobianTemp;
            
            NonZeroJacobianIndicesType nzjiTemp( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices());
            
            DerivativeType imageJacobianTemp( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
            
            unsigned int                            numSamplesOk = 0;
            std::vector<double>                     imageValues(this->m_NumSamplesLastDimension);
            std::vector<unsigned int >              positions(this->m_NumSamplesLastDimension);
            std::vector<NonZeroJacobianIndicesType> nzjis(this->m_NumSamplesLastDimension);
            std::vector<DerivativeType>             imageJacobians(this->m_NumSamplesLastDimension);
            
            for( unsigned int s = 0; s < this->m_NumSamplesLastDimension; s++ )
            {
                MovingImagePointType                    mappedPoint;
                RealType                                movingImageValueTemp;
                MovingImageDerivativeType               movingImageDerivativeTemp;
                
                voxelCoord[ lastDim ] = this->m_RandomList[ s ];
                
                this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord, fixedPoint );
                
                bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );
                
                if( sampleOk )
                {
                    sampleOk = this->IsInsideMovingMask( mappedPoint );
                }
                
                if( sampleOk )
                {
                    sampleOk = this->EvaluateMovingImageValueAndDerivative(mappedPoint, movingImageValueTemp, &movingImageDerivativeTemp );
                }
                if( sampleOk )
                {
                    
                    sampleOk = this->EvaluateTransformJacobian( fixedPoint, jacobianTemp, nzjiTemp);
                }
                if( sampleOk )
                {
                    this->EvaluateTransformJacobianInnerProduct(jacobianTemp, movingImageDerivativeTemp, imageJacobianTemp );
                    
                    if (movingImageValueTemp > this->m_FixedImageTrueMaxs[this->m_RandomList[ s ]])
                    {
                        imageValues[numSamplesOk] = this->m_FixedImageTrueMaxs[this->m_RandomList[ s ]];
                    }
                    else if (movingImageValueTemp < this->m_FixedImageTrueMins[this->m_RandomList[ s ]] )
                    {
                        imageValues[numSamplesOk] = this->m_FixedImageTrueMins[this->m_RandomList[ s ]];
                    }
                    else
                    {
                        imageValues[numSamplesOk] = movingImageValueTemp;
                    }
                    positions[numSamplesOk] = this->m_RandomList[ s ];
                    imageJacobians[numSamplesOk] = imageJacobianTemp;
                    nzjis[numSamplesOk] = nzjiTemp;
                    
                    numSamplesOk++;
                }
                
            }
            
            if ( numSamplesOk > 1)
            {
                this->m_NumberOfPixelsCounted++;
                imageValues.resize(numSamplesOk);
                positions.resize(numSamplesOk);
                imageJacobians.resize(numSamplesOk);
                nzjis.resize(numSamplesOk);
                
                for( unsigned int o = 0; o < numSamplesOk; o++ )
                {
                    this->m_NumberOfPixelsCountedVector[positions[o]]++;
                }
                
                this->UpdateJointPDFAndDerivatives(imageValues, &this->m_JointPDFs, &imageJacobians, &nzjis, numSamplesOk, positions);
            }
            
        }
        
        this->CheckNumberOfSamples( sampleContainer->Size(), this->m_NumberOfPixelsCounted );
    }
    
    /**
     * ******************** ComputeDerivativeLowMemory *******************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::ComputeDerivativeLowMemory( DerivativeType & derivative) const
    {
        if( !this->m_UseMultiThread )
        {
            return this->ComputeDerivativeLowMemorySingleThreaded( derivative);
        }
        
        this->LaunchComputeDerivativeLowMemoryThreaderCallback();
        
        this->AfterThreadedComputeDerivativeLowMemory( derivative );

    }
    
    
    /**
     * **************** ComputeDerivativeLowMemoryThreaderCallback *******
     */
    
    template< class TFixedImage, class TMovingImage >
    ITK_THREAD_RETURN_TYPE
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::ComputeDerivativeLowMemoryThreaderCallback( void * arg )
    {
        ThreadInfoType * infoStruct = static_cast< ThreadInfoType * >( arg );
        ThreadIdType     threadId   = infoStruct->ThreadID;
        
        ProbabilityAverageMutualInformationMetricMultiThreaderParameterType * temp
        = static_cast< ProbabilityAverageMutualInformationMetricMultiThreaderParameterType * >( infoStruct->UserData );
        
        temp->m_Metric->ThreadedComputeDerivativeLowMemory( threadId );
        
        return ITK_THREAD_RETURN_VALUE;
    }
    
    
    /**
     * *********************** LaunchComputeDerivativeLowMemoryThreaderCallback***************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::LaunchComputeDerivativeLowMemoryThreaderCallback( void ) const
    {
        typename ThreaderType::Pointer local_threader = ThreaderType::New();
        local_threader->SetNumberOfThreads( this->m_NumberOfThreads );
        local_threader->SetSingleMethod( this->ComputeDerivativeLowMemoryThreaderCallback, const_cast< void * >( static_cast< const void * >( &this->m_ProbabilityAverageMutualInformationMetricThreaderParameters ) ) );
        
        local_threader->SingleMethodExecute();
        
    }
    
    /**
     * *********************** ComputeDerivativeLowMemorySingleThreaded***************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::ComputeDerivativeLowMemorySingleThreaded( DerivativeType & derivative) const
    {
        const unsigned int lastDim     = this->GetFixedImage()->GetImageDimension() - 1;
        const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );
        
        typedef typename DerivativeType::ValueType        DerivativeValueType;
        typedef typename TransformJacobianType::ValueType TransformJacobianValueType;
        
        ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
        
        typename ImageSampleContainerType::ConstIterator fiter;
        typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
        typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();
        
        for( fiter = fbegin; fiter != fend; ++fiter )
        {
            FixedImagePointType fixedPoint = ( *fiter ).Value().m_ImageCoordinates;
            FixedImageContinuousIndexType       voxelCoord;
            this->GetFixedImage()->TransformPhysicalPointToContinuousIndex( fixedPoint, voxelCoord );
            
            TransformJacobianType  jacobianTemp;
            
            NonZeroJacobianIndicesType nzjiTemp( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices());
            
            DerivativeType imageJacobianTemp( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
            
            unsigned int                            numSamplesOk = 0;
            std::vector<double>                     imageValues(this->m_NumSamplesLastDimension);
            std::vector<unsigned int >              positions(this->m_NumSamplesLastDimension);
            std::vector<NonZeroJacobianIndicesType> nzjis(this->m_NumSamplesLastDimension);
            std::vector<DerivativeType>             imageJacobians(this->m_NumSamplesLastDimension);
            
            for( unsigned int s = 0; s < this->m_NumSamplesLastDimension; s++ )
            {
                MovingImagePointType                    mappedPoint;
                RealType                                movingImageValueTemp;
                MovingImageDerivativeType               movingImageDerivativeTemp;
                
                voxelCoord[ lastDim ] = this->m_RandomList[ s ];
                
                this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord, fixedPoint );
                
                bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );
                
                if( sampleOk )
                {
                    sampleOk = this->IsInsideMovingMask( mappedPoint );
                }
                
                if( sampleOk )
                {
                    sampleOk = this->EvaluateMovingImageValueAndDerivative(mappedPoint, movingImageValueTemp, &movingImageDerivativeTemp );
                }
                if( sampleOk )
                {
                    
                    sampleOk = this->EvaluateTransformJacobian( fixedPoint, jacobianTemp, nzjiTemp);
                }
                if( sampleOk )
                {
                    this->EvaluateTransformJacobianInnerProduct(jacobianTemp, movingImageDerivativeTemp, imageJacobianTemp );
                    
                    if (movingImageValueTemp > this->m_FixedImageTrueMaxs[this->m_RandomList[ s ]])
                    {
                        imageValues[numSamplesOk] = this->m_FixedImageTrueMaxs[this->m_RandomList[ s ]];
                    }
                    else if (movingImageValueTemp < this->m_FixedImageTrueMins[this->m_RandomList[ s ]] )
                    {
                        imageValues[numSamplesOk] = this->m_FixedImageTrueMins[this->m_RandomList[ s ]];
                    }
                    else
                    {
                        imageValues[numSamplesOk] = movingImageValueTemp;
                    }
                    positions[numSamplesOk] = this->m_RandomList[ s ];
                    imageJacobians[numSamplesOk] = imageJacobianTemp;
                    nzjis[numSamplesOk] = nzjiTemp;
                    
                    numSamplesOk++;
                }
                
            }
            
            if ( numSamplesOk > 1)
            {
                imageValues.resize(numSamplesOk);
                positions.resize(numSamplesOk);
                imageJacobians.resize(numSamplesOk);
                nzjis.resize(numSamplesOk);
                
                this->UpdateDerivativeLowMemory(imageValues, &imageJacobians, &nzjis, numSamplesOk, positions, derivative);
            }
            
        }
        
    }
    
    /**
     * *********************** ThreadedComputeDerivativeLowMemory***************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::ThreadedComputeDerivativeLowMemory( ThreadIdType threadId )
    {
        const unsigned int lastDim     = this->GetFixedImage()->GetImageDimension() - 1;
        const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );
        
        DerivativeType & derivative = this->m_GetValueAndDerivativePerThreadVariables[ threadId ].st_Derivative;

        derivative.Fill(0.0);

        typedef typename DerivativeType::ValueType        DerivativeValueType;
        typedef typename TransformJacobianType::ValueType TransformJacobianValueType;
        
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
            FixedImageContinuousIndexType       voxelCoord;
            this->GetFixedImage()->TransformPhysicalPointToContinuousIndex( fixedPoint, voxelCoord );
            
            TransformJacobianType  jacobianTemp;
            
            NonZeroJacobianIndicesType nzjiTemp( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices());
            
            DerivativeType imageJacobianTemp( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
            
            unsigned int                            numSamplesOk = 0;
            std::vector<double>                     imageValues(this->m_NumSamplesLastDimension);
            std::vector<unsigned int >              positions(this->m_NumSamplesLastDimension);
            std::vector<NonZeroJacobianIndicesType> nzjis(this->m_NumSamplesLastDimension);
            std::vector<DerivativeType>             imageJacobians(this->m_NumSamplesLastDimension);
            
            for( unsigned int s = 0; s < this->m_NumSamplesLastDimension; s++ )
            {
                MovingImagePointType                    mappedPoint;
                RealType                                movingImageValueTemp;
                MovingImageDerivativeType               movingImageDerivativeTemp;
                
                voxelCoord[ lastDim ] = this->m_RandomList[ s ];
                
                this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord, fixedPoint );
                
                bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );
                
                if( sampleOk )
                {
                    sampleOk = this->IsInsideMovingMask( mappedPoint );
                }
                
                if( sampleOk )
                {
                    sampleOk = this->EvaluateMovingImageValueAndDerivative(mappedPoint, movingImageValueTemp, &movingImageDerivativeTemp );
                }
                if( sampleOk )
                {
                    
                    sampleOk = this->EvaluateTransformJacobian( fixedPoint, jacobianTemp, nzjiTemp);
                }
                if( sampleOk )
                {
                    this->EvaluateTransformJacobianInnerProduct(jacobianTemp, movingImageDerivativeTemp, imageJacobianTemp );
                    
                    if (movingImageValueTemp > this->m_FixedImageTrueMaxs[this->m_RandomList[ s ]])
                    {
                        imageValues[numSamplesOk] = this->m_FixedImageTrueMaxs[this->m_RandomList[ s ]];
                    }
                    else if (movingImageValueTemp < this->m_FixedImageTrueMins[this->m_RandomList[ s ]] )
                    {
                        imageValues[numSamplesOk] = this->m_FixedImageTrueMins[this->m_RandomList[ s ]];
                    }
                    else
                    {
                        imageValues[numSamplesOk] = movingImageValueTemp;
                    }
                    positions[numSamplesOk] = this->m_RandomList[ s ];
                    imageJacobians[numSamplesOk] = imageJacobianTemp;
                    nzjis[numSamplesOk] = nzjiTemp;
                    
                    numSamplesOk++;
                }
                
            }
            
            if ( numSamplesOk > 1)
            {
                imageValues.resize(numSamplesOk);
                positions.resize(numSamplesOk);
                imageJacobians.resize(numSamplesOk);
                nzjis.resize(numSamplesOk);
                
                this->UpdateDerivativeLowMemory(imageValues, &imageJacobians, &nzjis, numSamplesOk, positions, derivative);
            }
            
        }

    }

    /**
     * ******************* AfterThreadedComputeDerivativeLowMemory *******************
     */
    
    
    template< class TFixedImage, class TMovingImage >
    void
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::AfterThreadedComputeDerivativeLowMemory( DerivativeType & derivative ) const
    {
        for( ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i )
        {
            DerivativeType & derivativeTemp = this->m_GetValueAndDerivativePerThreadVariables[ 0 ].st_Derivative;
            
            derivative += derivativeTemp;
            
        }
    }
    
    /**
     * ********************** UpdateJointPDFAndDerivatives ***************
     */
    template< class TFixedImage, class TMovingImage >
    void
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::UpdateJointPDFAndDerivatives(const std::vector<double> & fixedImageValue, const std::vector<JointPDFPointer> * jointPDFs, const std::vector<DerivativeType> * imageJacobian, const std::vector<NonZeroJacobianIndicesType> * nzji, const unsigned int n, const std::vector<unsigned int> & positions  ) const
    {
        std::vector<ParzenValueContainerType> fixedParzenValues(n);
        std::vector<ParzenValueContainerType> derivativeFixedParzenValues(n);

        std::vector<OffsetValueType> fixedImageParzenWindowIndexs(n);

        for (unsigned int o = 0; o < n; o++)
        {
            fixedParzenValues[o] = ParzenValueContainerType(this->m_JointPDFWindow.GetSize()[ 1 ]);
            
            const double fixedImageParzenWindowTerm = fixedImageValue[o] / this->m_FixedImageBinSizes[positions[o]] - this->m_FixedImageNormalizedMins[positions[o]];
            fixedImageParzenWindowIndexs[o] = static_cast< OffsetValueType >( vcl_floor(fixedImageParzenWindowTerm + this->m_FixedParzenTermToIndexOffset ) );
            
            this->EvaluateParzenValues( fixedImageParzenWindowTerm, fixedImageParzenWindowIndexs[o], this->m_FixedKernel, fixedParzenValues[o] );
            
            if( !imageJacobian )
            {
                derivativeFixedParzenValues.resize(0);
            }
            else
            {
                derivativeFixedParzenValues[o] = ParzenValueContainerType(this->m_JointPDFWindow.GetSize()[ 1 ]);
                this->EvaluateParzenValues( fixedImageParzenWindowTerm, fixedImageParzenWindowIndexs[o], this->m_DerivativeMovingKernel, derivativeFixedParzenValues[o] );

            }
            
        }
        
        JointPDFIndexType pdfWindowIndex;
        typedef ImageScanlineIterator< JointPDFType > PDFIteratorType;

        for (unsigned int o = 0; o < n; o++)
        {
            if (fixedImageParzenWindowIndexs[o] > (this->m_NumberOfFixedHistogramBins - this->m_FixedKernelBSplineOrder -1 ))
            {
                std::cout << "Warning IOFB in UJaD: FIPWI: " << fixedImageValue[o] << " " << this->m_FixedImageBinSizes[positions[o]] << " "<< this->m_FixedImageNormalizedMins[positions[o]] << " " << fixedImageParzenWindowIndexs[o] <<  " " << o << " " << positions[o] << std::endl;
                
                pdfWindowIndex[ 1 ] = (this->m_NumberOfFixedHistogramBins - this->m_FixedKernelBSplineOrder -1 );
            }
            else if (fixedImageParzenWindowIndexs[o] < 0 )
            {
                std::cout << "Warning IOFB in UJaD: MIPWI: " << fixedImageValue[o] << " " << this->m_FixedImageBinSizes[positions[o]] << " "<< this->m_FixedImageNormalizedMins[positions[o]] << " " << fixedImageParzenWindowIndexs[o] <<  " " << o << " " << positions[o] << std::endl;
                pdfWindowIndex[ 1 ] = 0;
            }
            else
            {
                pdfWindowIndex[ 1 ] = fixedImageParzenWindowIndexs[o];
            }
            
            for (unsigned int k = 0; k < n; k++)
            {
                if( o != k )
                {
                    ParzenValueContainerType movingParzenValues = ParzenValueContainerType(this->m_JointPDFWindow.GetSize()[ 0 ]);

                    const double movingImageParzenWindowTerm = fixedImageValue[k] / this->m_MovingImageBinSizes[positions[o]] - this->m_MovingImageNormalizedMins[positions[o]];
                
                    const OffsetValueType movingImageParzenWindowIndex = static_cast< OffsetValueType >( vcl_floor(movingImageParzenWindowTerm + this->m_MovingParzenTermToIndexOffset ) );
                
                    this->EvaluateParzenValues(movingImageParzenWindowTerm, movingImageParzenWindowIndex, this->m_MovingKernel, movingParzenValues );

                    
                    if (movingImageParzenWindowIndex > (this->m_NumberOfMovingHistogramBins - this->m_MovingKernelBSplineOrder -1))
                    {
                        std::cout << "Warning IOFB in UJaD: MIPWI: " << fixedImageValue[k]  << "\t" << this->m_MovingImageBinSizes[positions[o]]  << "\t" << this->m_MovingImageNormalizedMins[positions[o]] << "\t" << movingImageParzenWindowIndex << " " << k << " " << positions[k] << std::endl;
                        pdfWindowIndex[ 0 ] = (this->m_NumberOfMovingHistogramBins - this->m_MovingKernelBSplineOrder -1);
                        
                    }
                    else if (movingImageParzenWindowIndex < 0)
                    {
                        std::cout << "Warning IOFB in UJaD: MIPWI: " << fixedImageValue[k]  << "\t" << this->m_MovingImageBinSizes[positions[o]]  << "\t" << this->m_MovingImageNormalizedMins[positions[o]] << "\t" << movingImageParzenWindowIndex << " " << k << " " << positions[k] << std::endl;
                        pdfWindowIndex[ 0 ] = 0;
                    }
                    else
                    {
                        pdfWindowIndex[ 0 ] = movingImageParzenWindowIndex;
                    }
                    
                    JointPDFRegionType jointPDFWindow = this->m_JointPDFWindow;
                    jointPDFWindow.SetIndex( pdfWindowIndex );
                    PDFIteratorType it( (*jointPDFs)[positions[o]], jointPDFWindow );
                        
                    if( !imageJacobian )
                    {
                        for( unsigned int f = 0; f < fixedParzenValues[o].GetSize(); f++ )
                        {
                            const double fv = fixedParzenValues[o][ f ];
                            for( unsigned int m = 0; m < movingParzenValues.GetSize(); m++ )
                            {
                                const double mv = movingParzenValues[m];
                                it.Value() += static_cast< PDFValueType >( fv * mv / static_cast<double>(n-1));
                                ++it;
                            }
                            it.NextLine();
                        }
                    }
                    
                    else
                    {
                        ParzenValueContainerType derivativeMovingParzenValues(this->m_JointPDFWindow.GetSize()[ 0 ] );
                        this->EvaluateParzenValues( movingImageParzenWindowTerm, movingImageParzenWindowIndex, this->m_DerivativeMovingKernel, derivativeMovingParzenValues );
                        const double etm = static_cast< double >( this->m_MovingImageBinSizes[positions[o]] );
                        const double etf = static_cast< double >( this->m_FixedImageBinSizes[positions[o]] );
                        for( unsigned int f = 0; f < fixedParzenValues[o].GetSize(); ++f )
                        {
                            const double fv    = fixedParzenValues[o][f];
                            const double fv_etm = fv / etm;
                                
                            for( unsigned int m = 0; m < movingParzenValues.GetSize(); ++m )
                            {
                                const double mv = movingParzenValues[m];
                                const double mv_etf = mv / etf;
                                it.Value() += static_cast< PDFValueType >( fv * mv / static_cast<double>(n-1) );
                                this->UpdateJointPDFDerivatives(it.GetIndex(), fv_etm * derivativeMovingParzenValues[ m ], mv_etf * derivativeFixedParzenValues[o][f], (*imageJacobian)[o], (*nzji)[o], (*imageJacobian)[k], (*nzji)[k], o, positions, n);
                                ++it;
                            }
                            it.NextLine();
                        }

                    }
                    
                }

            }
            
        }
        
    }
    
    /**
     * *************** UpdateJointPDFDerivatives ***************************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::UpdateJointPDFDerivatives( const JointPDFIndexType & pdfIndex, double factor, double factorr, const DerivativeType & imageJacobianFix, const NonZeroJacobianIndicesType & nzjiFix, const DerivativeType & imageJacobianMov, const NonZeroJacobianIndicesType & nzjiMov, const unsigned int o, const std::vector<unsigned int> & positions, const unsigned int n) const
    {
        PDFDerivativeValueType * derivPtr = this->m_JointPDFDerivatives[positions[o]]->GetBufferPointer() + ( pdfIndex[ 0 ] * this->m_JointPDFDerivatives[positions[o]]->GetOffsetTable()[ 1 ] ) + ( pdfIndex[ 1 ] * this->m_JointPDFDerivatives[positions[o]]->GetOffsetTable()[ 2 ] );
        typename DerivativeType::const_iterator imjac = imageJacobianFix.begin();
        
        for ( unsigned int i = 0; i < nzjiFix.size(); i++)
        {
            const unsigned int       mu  = nzjiFix[i];
            PDFDerivativeValueType * ptr = derivPtr + mu;
            
            *( ptr ) -= static_cast< PDFDerivativeValueType >( (*imjac) * (factorr / static_cast<double>(n - 1) ) );
            imjac++;
            
        }
        
        imjac = imageJacobianMov.begin();
        
        for ( unsigned int i = 0; i < nzjiMov.size(); i++)
        {
            const unsigned int       mu  = nzjiMov[i];
            PDFDerivativeValueType * ptr = derivPtr + mu;
            
            *( ptr ) -= static_cast< PDFDerivativeValueType >( (*imjac) * (factor / static_cast<double>(n - 1) ) );
            imjac++;
            
        }
        
    }
    
    /**
     * ******************* UpdateDerivativeLowMemory *******************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::UpdateDerivativeLowMemory( const std::vector<double> & fixedImageValue, const std::vector<DerivativeType> * imageJacobian, const std::vector<NonZeroJacobianIndicesType> * nzji, const unsigned int n, const std::vector<unsigned int> & positions,DerivativeType & derivative) const
    {
        std::vector<ParzenValueContainerType> fixedParzenValues(n);
        std::vector<ParzenValueContainerType> derivativeFixedParzenValues(n);
        
        std::vector<OffsetValueType> fixedImageParzenWindowIndexs(n);
        
        for (unsigned int o = 0; o < n; o++)
        {
            fixedParzenValues[o] = ParzenValueContainerType(this->m_JointPDFWindow.GetSize()[ 1 ]);
            
            const double fixedImageParzenWindowTerm = fixedImageValue[o] / this->m_FixedImageBinSizes[positions[o]] - this->m_FixedImageNormalizedMins[positions[o]];
            fixedImageParzenWindowIndexs[o] = static_cast< OffsetValueType >( vcl_floor(fixedImageParzenWindowTerm + this->m_FixedParzenTermToIndexOffset ) );
            
            this->EvaluateParzenValues( fixedImageParzenWindowTerm, fixedImageParzenWindowIndexs[o], this->m_FixedKernel, fixedParzenValues[o] );
            
            derivativeFixedParzenValues[o] = ParzenValueContainerType(this->m_JointPDFWindow.GetSize()[ 1 ]);
            this->EvaluateParzenValues( fixedImageParzenWindowTerm, fixedImageParzenWindowIndexs[o], this->m_DerivativeMovingKernel, derivativeFixedParzenValues[o] );
            
        }
        
        for (unsigned int o = 0; o < n; o++)
        {
            if (fixedImageParzenWindowIndexs[o] > (this->m_NumberOfFixedHistogramBins - this->m_FixedKernelBSplineOrder -1 ))
            {
                std::cout << "Warning IOFB in UDLM: FIPWI: " << fixedImageValue[o] << " " << this->m_FixedImageBinSizes[positions[o]] << " "<< this->m_FixedImageNormalizedMins[positions[o]] << " " << fixedImageParzenWindowIndexs[o] <<  " " << o << " " << positions[o] << std::endl;
                
                fixedImageParzenWindowIndexs[o] = (this->m_NumberOfFixedHistogramBins - this->m_FixedKernelBSplineOrder -1 );
            }
            else if (fixedImageParzenWindowIndexs[o] < 0 )
            {
                std::cout << "Warning IOFB in UDLM: MIPWI: " << fixedImageValue[o] << " " << this->m_FixedImageBinSizes[positions[o]] << " "<< this->m_FixedImageNormalizedMins[positions[o]] << " " << fixedImageParzenWindowIndexs[o] <<  " " << o << " " << positions[o] << std::endl;
                fixedImageParzenWindowIndexs[o] = 0;
            }
            
            for (unsigned int k = 0; k < n; k++)
            {
                if( o != k )
                {
                    ParzenValueContainerType movingParzenValues = ParzenValueContainerType(this->m_JointPDFWindow.GetSize()[ 0 ]);
                    
                    const double movingImageParzenWindowTerm = fixedImageValue[k] / this->m_MovingImageBinSizes[positions[o]] - this->m_MovingImageNormalizedMins[positions[o]];
                    
                    OffsetValueType movingImageParzenWindowIndex = static_cast< OffsetValueType >( vcl_floor(movingImageParzenWindowTerm + this->m_MovingParzenTermToIndexOffset ) );
                    
                    this->EvaluateParzenValues(movingImageParzenWindowTerm, movingImageParzenWindowIndex, this->m_MovingKernel, movingParzenValues );

                    if (movingImageParzenWindowIndex > (this->m_NumberOfMovingHistogramBins - this->m_MovingKernelBSplineOrder -1))
                    {
                        std::cout << "Warning IOFB in UDLM: MIPWI: " << fixedImageValue[k]  << "\t" << this->m_MovingImageBinSizes[positions[o]]  << "\t" << this->m_MovingImageNormalizedMins[positions[o]] << "\t" << movingImageParzenWindowIndex << " " << k << " " << positions[k] << std::endl;
                        movingImageParzenWindowIndex = (this->m_NumberOfMovingHistogramBins - this->m_MovingKernelBSplineOrder -1);
                        
                    }
                    if (movingImageParzenWindowIndex < 0)
                    {
                        std::cout << "Warning IOFB in UDLM: MIPWI: " << fixedImageValue[k]  << "\t" << this->m_MovingImageBinSizes[positions[o]]  << "\t" << this->m_MovingImageNormalizedMins[positions[o]] << "\t" << movingImageParzenWindowIndex << " " << k << " " << positions[k] << std::endl;
                        movingImageParzenWindowIndex = 0;
                    }

                    ParzenValueContainerType derivativeMovingParzenValues(this->m_JointPDFWindow.GetSize()[ 0 ] );
                    this->EvaluateParzenValues( movingImageParzenWindowTerm, movingImageParzenWindowIndex, this->m_DerivativeMovingKernel, derivativeMovingParzenValues );

                    const double etm = static_cast< double >( this->m_MovingImageBinSizes[positions[o]] );
                    const double etf = static_cast< double >( this->m_FixedImageBinSizes[positions[o]]);
                        
                    double regSum = 0.0;
                    double extraSum = 0.0;
                        
                    for( unsigned int f = 0; f < fixedParzenValues[o].GetSize(); ++f )
                    {
                        const double fv    = fixedParzenValues[o][f];
                        const double fv_etm = fv / etm;
                        for( unsigned int m = 0; m < movingParzenValues.GetSize(); ++m )
                        {
                            const double mv = movingParzenValues[m];
                            const double mv_etf = mv / etf;
                                
                            extraSum = extraSum + (*(this->m_PRatioArray[positions[o]]))[ f + fixedImageParzenWindowIndexs[o] ][ m + movingImageParzenWindowIndex ] * mv_etf * derivativeFixedParzenValues[o][ f ] ;
                                
                            regSum = regSum + (*(this->m_PRatioArray[positions[o]]))[ f + fixedImageParzenWindowIndexs[o] ][ m + movingImageParzenWindowIndex ] * fv_etm * derivativeMovingParzenValues[ m ];
                                
                        }
                            
                    }
                        
                    typename DerivativeType::const_iterator imjac = (*imageJacobian)[o].begin();
                    
                    for ( unsigned int i = 0; i < (*nzji)[o].size(); i++)
                    {
                        const unsigned int       mu  = (*nzji)[o][i];
                            
                        derivative[ mu ] += static_cast< PDFDerivativeValueType >( (*imjac) * (extraSum / static_cast<double>(n - 1) ) );
                        imjac++;
                            
                    }
                        
                    imjac = (*imageJacobian)[k].begin();
                        
                    for ( unsigned int i = 0; i < (*nzji)[k].size(); i++)
                    {
                        const unsigned int       mu  = (*nzji)[k][i];
                            
                        derivative[ mu ] += static_cast< PDFDerivativeValueType >( (*imjac) * (regSum / static_cast<double>(n - 1) ) );
                            imjac++;
                            
                    }
                    
                }
                
            }
            
        }
        
    }
}

#endif // end #ifndef _itkProbabilityAverageMutualInformationMetric_HXX__
