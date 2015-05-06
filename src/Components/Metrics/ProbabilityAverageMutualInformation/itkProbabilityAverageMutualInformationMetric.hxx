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
        this->m_TransformIsStackTransform = false;
        
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
        for(unsigned int n = 0; n < this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension); n++)
        {
            FixedImagePixelType trueMinTemp = NumericTraits< FixedImagePixelType >::max();
            FixedImagePixelType trueMaxTemp = NumericTraits< FixedImagePixelType >::NonpositiveMin();
            
            /** If no mask. */
            if( this->m_FixedImageMask.IsNull() )
            {
                typedef ImageRegionConstIterator< FixedImageType > IteratorType;
                
                FixedImageRegionType region = this->GetFixedImage()->GetLargestPossibleRegion();
                region.SetIndex(ReducedFixedImageDimension, n);
                region.SetSize(ReducedFixedImageDimension, 1);
                IteratorType it(this->GetFixedImage(), region );
                
                for( it.GoToBegin(); !it.IsAtEnd(); ++it )
                {
                    const FixedImagePixelType sample = it.Get();
                    trueMinTemp = vnl_math_min( trueMinTemp, sample );
                    trueMaxTemp = vnl_math_max( trueMaxTemp, sample );
                    
                }
            }
            
            else
            {
                typedef ImageRegionConstIteratorWithIndex< FixedImageType > IteratorType;
                
                FixedImageRegionType region = this->GetFixedImage()->GetLargestPossibleRegion();
                
                region.SetIndex(ReducedFixedImageDimension, n);
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
                        trueMaxTemp = vnl_math_max( trueMaxTemp, sample );
                        
                    }
                    
                }
            }
            
            this->m_FixedImageTrueMins[n] = trueMinTemp;
            this->m_FixedImageTrueMaxs[n] = trueMaxTemp;
            
            this->m_FixedImageMinLimits[n] = static_cast< FixedImageLimiterOutputType >(trueMinTemp - 0.01 * ( trueMaxTemp - trueMinTemp ) );
            this->m_FixedImageMaxLimits[n] =  static_cast< FixedImageLimiterOutputType >(trueMaxTemp + 0.01 * ( trueMaxTemp - trueMinTemp ) );

        }
        
        for(unsigned int n = 0; n < this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension); n++)
        {
            MovingImagePixelType trueMinTemp = NumericTraits< FixedImagePixelType >::max();
            MovingImagePixelType trueMaxTemp = NumericTraits< FixedImagePixelType >::NonpositiveMin();
            
            for(unsigned int m = 0; m < this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension); m++)
            {
                if ( m != n )
                {
                    trueMinTemp = vnl_math_min( trueMinTemp, this->m_FixedImageTrueMins[m] );
                    trueMaxTemp = vnl_math_max( trueMaxTemp, this->m_FixedImageTrueMaxs[m] );
                }
            }
            
            
            this->m_MovingImageTrueMins[n] = trueMinTemp;
            this->m_MovingImageTrueMaxs[n] = trueMaxTemp;

            this->m_MovingImageMinLimits[n] = static_cast< MovingImageLimiterOutputType >( this->m_MovingImageTrueMins[n] - 0.01 * ( this->m_MovingImageTrueMaxs[n] - this->m_MovingImageTrueMins[n] ));
            this->m_MovingImageMaxLimits[n] = static_cast< MovingImageLimiterOutputType >( this->m_MovingImageTrueMaxs[n] + 0.01 * ( this->m_MovingImageTrueMaxs[n] - this->m_MovingImageTrueMins[n] ));

        }
        
        
        for(unsigned int n = 0; n < this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension); n++)
        {
            this->m_FixedImageMarginalPDFs[n] = new MarginalPDFType;
            this->m_FixedImageMarginalPDFs[n]->SetSize(this->m_NumberOfFixedHistogramBins);
            this->m_MovingImageMarginalPDFs[n] = new MarginalPDFType;
            this->m_MovingImageMarginalPDFs[n]->SetSize( this->m_NumberOfMovingHistogramBins );
            this->m_JointPDFs[n] = JointPDFType::New();
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
        int fixedPadding  = this->m_FixedKernelBSplineOrder / 2;
        int movingPadding = this->m_MovingKernelBSplineOrder / 2;
        
        const double smallNumberRatio = 0.001;
        
        for(unsigned int n = 0; n < this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension); n++)
        {
            const double fixedHistogramWidth = static_cast< double >( static_cast< OffsetValueType >( this->m_NumberOfFixedHistogramBins ) - 2.0 * fixedPadding - 1.0 );
            const double movingHistogramWidth = static_cast< double >( static_cast< OffsetValueType >( this->m_NumberOfMovingHistogramBins ) - 2.0 * movingPadding - 1.0 );

            double smallNumberMoving = smallNumberRatio * ( this->m_MovingImageMaxLimits[n] - this->m_MovingImageMinLimits[n] ) / static_cast< double >( this->m_NumberOfMovingHistogramBins - 2 * movingPadding - 1 );
            
            this->m_MovingImageBinSizes[n] = ( this->m_MovingImageMaxLimits[n] - this->m_MovingImageMinLimits[n] + 2.0 * smallNumberMoving ) / movingHistogramWidth;
            
            this->m_MovingImageBinSizes[n] = vnl_math_max( this->m_MovingImageBinSizes[n], 1e-10 );
            this->m_MovingImageBinSizes[n] = vnl_math_min( this->m_MovingImageBinSizes[n], 1e+10 );
            this->m_MovingImageNormalizedMins[n] = ( this->m_MovingImageMinLimits[n] - smallNumberMoving ) / this->m_MovingImageBinSizes[n] - static_cast< double >( movingPadding );
            
            const double smallNumberFixed = smallNumberRatio * ( this->m_FixedImageMaxLimits[n] - this->m_FixedImageMinLimits[n] ) / static_cast< double >( this->m_NumberOfFixedHistogramBins - 2 * fixedPadding - 1 );
            this->m_FixedImageBinSizes[n] = static_cast<double>( this->m_FixedImageMaxLimits[n] - this->m_FixedImageMinLimits[n]+ 2.0 * smallNumberFixed ) / fixedHistogramWidth;
            
            this->m_FixedImageBinSizes[n] = vnl_math_max( this->m_FixedImageBinSizes[n], 1e-10 );
            this->m_FixedImageBinSizes[n] = vnl_math_min( this->m_FixedImageBinSizes[n], 1e+10 );
            this->m_FixedImageNormalizedMins[n] = (this->m_FixedImageMinLimits[n] - smallNumberFixed ) / this->m_FixedImageBinSizes[n] - static_cast< double >( fixedPadding );
            
        }
        
        JointPDFRegionType jointPDFRegion;
        JointPDFIndexType  jointPDFIndex;
        JointPDFSizeType   jointPDFSize;
        jointPDFIndex.Fill( 0 );
        jointPDFSize[ 0 ] = this->m_NumberOfMovingHistogramBins;
        jointPDFSize[ 1 ] = this->m_NumberOfFixedHistogramBins;
        jointPDFRegion.SetIndex( jointPDFIndex );
        jointPDFRegion.SetSize( jointPDFSize );
        
        for(unsigned int n = 0; n < this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension); n++)
        {
            this->m_JointPDFs[n]->SetRegions( jointPDFRegion );
            this->m_JointPDFs[n]->Allocate();
            this->m_JointPDFs[n]->FillBuffer(0.0);
            
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
                    this->m_JointPDFDerivatives[n] = JointPDFDerivativesType::New();
                    this->m_JointPDFDerivatives[n]->SetRegions( jointPDFDerivativesRegion );
                    this->m_JointPDFDerivatives[n]->Allocate();
                    this->m_JointPDFDerivatives[n]->FillBuffer(0.0);
                }
                
                else
                {
                    if( !this->m_JointPDFDerivatives[n].IsNull() )
                    {
                        jointPDFDerivativesSize.Fill( 0 );
                        jointPDFDerivativesRegion.SetSize( jointPDFDerivativesSize );
                        this->m_JointPDFDerivatives[n]->SetRegions( jointPDFDerivativesRegion );
                        this->m_JointPDFDerivatives[n]->Allocate();
                        this->m_JointPDFDerivatives[n]->GetPixelContainer()->Squeeze();
                    }
                }
                
            }
            
            else
            {
                this->m_JointPDFDerivatives[n]  = 0;				
            }	
            
        }
        
        if( !this->GetUseExplicitPDFDerivatives() )
        {
            this->m_PRatioArray.resize(this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension));
            
            for(unsigned int m = 0; m < this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension); m++)
            {
                this->m_PRatioArray[m] = new PRatioArrayType;
                this->m_PRatioArray[m]->SetSize( this->GetNumberOfFixedHistogramBins(), this->GetNumberOfMovingHistogramBins() );
                this->m_PRatioArray[m]->Fill( itk::NumericTraits< PRatioType >::Zero  );
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
        
        for(unsigned int n = 0; n < this->m_NumSamplesLastDimension; n++)
        {
            this->NormalizeJointPDF( this->m_JointPDFs[this->m_RandomList[n]], this->m_Alpha );
            this->ComputeMarginalPDF( this->m_JointPDFs[this->m_RandomList[n]], this->m_FixedImageMarginalPDFs[this->m_RandomList[n]], 0 );
            this->ComputeMarginalPDF( this->m_JointPDFs[this->m_RandomList[n]], this->m_MovingImageMarginalPDFs[this->m_RandomList[n]], 1 );
        }
        
        typedef ImageLinearConstIteratorWithIndex< JointPDFType >           JointPDFIteratorType;
        typedef ImageLinearConstIteratorWithIndex<JointPDFDerivativesType >JointPDFDerivativesIteratorType;
        typedef typename MarginalPDFType::const_iterator                    MarginalPDFIteratorType;
        
        this->m_Values.assign(this->m_NumSamplesLastDimension,0.0);
        
        for(unsigned int n = 0; n < this->m_NumSamplesLastDimension; n++)
        {
            JointPDFIteratorType jointPDFit(this->m_JointPDFs[this->m_RandomList[n]], this->m_JointPDFs[this->m_RandomList[n]]->GetLargestPossibleRegion() );
            jointPDFit.SetDirection( 0 );
            jointPDFit.GoToBegin();
            
            MarginalPDFIteratorType       fixedPDFit   = this->m_FixedImageMarginalPDFs[this->m_RandomList[n]]->begin();
            const MarginalPDFIteratorType fixedPDFend  = this->m_FixedImageMarginalPDFs[this->m_RandomList[n]]->end();
            MarginalPDFIteratorType       movingPDFit  = this->m_MovingImageMarginalPDFs[this->m_RandomList[n]]->begin();
            const MarginalPDFIteratorType movingPDFend = this->m_MovingImageMarginalPDFs[this->m_RandomList[n]]->end();
            
            double MI = 0.0;
            
            while( fixedPDFit != fixedPDFend )
            {
                const double fixedImagePDFValue = *fixedPDFit;
                movingPDFit = this->m_MovingImageMarginalPDFs[this->m_RandomList[n]]->begin();
                
                while( movingPDFit != movingPDFend )
                {
                    const double movingImagePDFValue = *movingPDFit;
                    const double fixPDFmovPDF        = fixedImagePDFValue * movingImagePDFValue;
                    const double jointPDFValue       = jointPDFit.Get();
                    
                    if( jointPDFValue > 1e-16 && fixPDFmovPDF > 1e-16 )
                    {
                        const double pRatio      = vcl_log( jointPDFValue / fixPDFmovPDF );
                        const double pRatioAlpha = this->m_Alpha * pRatio;
                        MI = MI + jointPDFValue * pRatio;
                    }
                    
                    ++movingPDFit;
                    ++jointPDFit;
                    
                }
                ++fixedPDFit;
                jointPDFit.NextLine();
                
            }
            this->m_Values[n] = MI;
        }
        
        double PAMI =0.0;
        for(unsigned int n = 0; n < this->m_NumSamplesLastDimension; n++)
        {
            PAMI = PAMI + this->m_Values[n];
        }
        
        return static_cast< MeasureType >( -1.0 * PAMI);

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
        
        for(unsigned int n = 0; n < this->m_NumSamplesLastDimension; n++)
        {
            this->NormalizeJointPDF( this->m_JointPDFs[this->m_RandomList[n]], this->m_Alpha );
            this->ComputeMarginalPDF( this->m_JointPDFs[this->m_RandomList[n]], this->m_FixedImageMarginalPDFs[this->m_RandomList[n]], 0 );
            this->ComputeMarginalPDF( this->m_JointPDFs[this->m_RandomList[n]], this->m_MovingImageMarginalPDFs[this->m_RandomList[n]], 1 );
        }
        
        typedef ImageLinearConstIteratorWithIndex< JointPDFType > JointPDFIteratorType;
        typedef ImageLinearConstIteratorWithIndex<JointPDFDerivativesType >                       JointPDFDerivativesIteratorType;
        typedef typename MarginalPDFType::const_iterator          MarginalPDFIteratorType;
        
        typedef typename DerivativeType::iterator        DerivativeIteratorType;
        typedef typename DerivativeType::const_iterator  DerivativeConstIteratorType;
        
        std::vector<DerivativeType> derivatives;
        derivatives.resize(this->m_NumSamplesLastDimension);
        this->m_Values.assign(this->m_NumSamplesLastDimension,0.0);
        
        
        for(unsigned int n = 0; n < this->m_NumSamplesLastDimension; n++)
        {
            JointPDFIteratorType jointPDFit(this->m_JointPDFs[this->m_RandomList[n]], this->m_JointPDFs[this->m_RandomList[n]]->GetLargestPossibleRegion() );
            jointPDFit.SetDirection( 0 );
            jointPDFit.GoToBegin();
            JointPDFDerivativesIteratorType jointPDFDerivativesit(this->m_JointPDFDerivatives[this->m_RandomList[n]], this->m_JointPDFDerivatives[this->m_RandomList[n]]->GetLargestPossibleRegion());
            jointPDFDerivativesit.SetDirection( 0 );
            jointPDFDerivativesit.GoToBegin();
            MarginalPDFIteratorType       fixedPDFit   = this->m_FixedImageMarginalPDFs[this->m_RandomList[n]]->begin();
            const MarginalPDFIteratorType fixedPDFend  = this->m_FixedImageMarginalPDFs[this->m_RandomList[n]]->end();
            MarginalPDFIteratorType       movingPDFit  = this->m_MovingImageMarginalPDFs[this->m_RandomList[n]]->begin();
            const MarginalPDFIteratorType movingPDFend = this->m_MovingImageMarginalPDFs[this->m_RandomList[n]]->end();
            
            derivative.Fill( NumericTraits< double >::Zero );
            DerivativeIteratorType        derivit      = derivative.begin();
            const DerivativeIteratorType  derivbegin   = derivative.begin();
            const DerivativeIteratorType  derivend     = derivative.end();
            
            double MI = 0.0;
            
            while( fixedPDFit != fixedPDFend )
            {
                const double fixedImagePDFValue = *fixedPDFit;
                movingPDFit = this->m_MovingImageMarginalPDFs[this->m_RandomList[n]]->begin();
                while( movingPDFit != movingPDFend )
                {
                    const double movingImagePDFValue = *movingPDFit;
                    const double fixPDFmovPDF        = fixedImagePDFValue * movingImagePDFValue;
                    const double jointPDFValue       = jointPDFit.Get();
                    
                    if( jointPDFValue > 1e-16 && fixPDFmovPDF > 1e-16 )
                    {
                        derivit = derivbegin;
                        const double pRatio      = vcl_log( jointPDFValue / fixPDFmovPDF );
                        const double pRatioAlpha = this->m_Alpha * pRatio;
                        MI = MI + jointPDFValue * pRatio;
                        while( derivit != derivend )
                        {
                            ( *derivit ) -= jointPDFDerivativesit.Get() * pRatioAlpha;
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
            
            this->m_Values[n] = MI;
            derivatives[n] = derivative;
            
        }
        
        derivative.Fill(0.0);
        double PAMI = 0.0;
        
        for(unsigned int n = 0; n < this->m_NumSamplesLastDimension; n++)
        {
            derivative = derivative + derivatives[n];
            PAMI = PAMI + this->m_Values[n];
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

        value = static_cast< MeasureType >( -1.0 * PAMI );
        

    }
    
    /**
     * *********************  GetValueAndAnalyticalDerivativeLowMemory ****************************
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
        
        for(unsigned int n = 0; n < this->m_NumSamplesLastDimension; n++)
        {
            this->NormalizeJointPDF( this->m_JointPDFs[this->m_RandomList[n]], this->m_Alpha );
            this->ComputeMarginalPDF( this->m_JointPDFs[this->m_RandomList[n]], this->m_FixedImageMarginalPDFs[this->m_RandomList[n]], 0 );
            this->ComputeMarginalPDF( this->m_JointPDFs[this->m_RandomList[n]], this->m_MovingImageMarginalPDFs[this->m_RandomList[n]], 1 );
            this->ComputeValueAndPRatioArray( MI, n);
            this->m_Values[n] = MI;
        }
        
        this->ComputeDerivativeLowMemory( derivative);
        
        PAMI = 0.0;
        
        for(unsigned int n = 0; n < this->m_NumSamplesLastDimension; n++)
        {
            PAMI = PAMI + this->m_Values[n];
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
        Superclass::InitializeThreadingParameters();
        
        if( this->m_ProbabilityAverageMutualInformationMetricGetValueAndDerivativePerThreadVariablesSize != this->m_NumberOfThreads )
        {
            delete[] this->m_ProbabilityAverageMutualInformationMetricGetValueAndDerivativePerThreadVariables;
            this->m_ProbabilityAverageMutualInformationMetricGetValueAndDerivativePerThreadVariables = new AlignedProbabilityAverageMutualInformationMetricGetValueAndDerivativePerThreadStruct[this->m_NumberOfThreads ];
            this->m_ProbabilityAverageMutualInformationMetricGetValueAndDerivativePerThreadVariablesSize = this->m_NumberOfThreads;
        }
        
        for( ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i )
        {
            this->m_ProbabilityAverageMutualInformationMetricGetValueAndDerivativePerThreadVariables[ i ].st_NumberOfPixelsCounted = NumericTraits< SizeValueType >::Zero;
        }

        for(unsigned int n = 0; n < this->m_NumSamplesLastDimension; n++)
        {
            this->m_JointPDFs[this->m_RandomList[n]]->FillBuffer( 0.0 );
            if (this->m_UseExplicitPDFDerivatives)
            {
                this->m_JointPDFDerivatives[this->m_RandomList[n]]->FillBuffer(0.0);
            }
        }

        this->m_Alpha= 0.0;
        
        if( !this->m_SampleLastDimensionRandomly )
        {
            for( unsigned int i = 0; i < this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension); i++ )
            {
                this->m_RandomList[i]=i;
            }
        }
        else
        {
            this->SampleRandom( this->m_NumSamplesLastDimension, this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension), this->m_RandomList );
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
        this->m_NumberOfPixelsCounted = this->m_ProbabilityAverageMutualInformationMetricGetValueAndDerivativePerThreadVariables[ 0 ].st_NumberOfPixelsCounted;
        
        ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
        this->CheckNumberOfSamples( sampleContainer->Size(), this->m_NumberOfPixelsCounted );
        
        this->m_Alpha = 1.0 / static_cast< double >(this->m_NumberOfPixelsCounted);

    }

    
    /**
     * ************************ ComputePDFsSingleThreaded **************************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::ComputePDFsSingleThreaded( const ParametersType & parameters ) const
    {
        
        for(unsigned int n = 0; n < this->m_NumSamplesLastDimension; n++)
        {
            this->m_JointPDFs[this->m_RandomList[n]]->FillBuffer( 0.0 );
        }
        
        this->m_Alpha = 0.0;
        this->m_NumberOfPixelsCounted = 0;

        this->BeforeThreadedGetValueAndDerivative( parameters );
        
        ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
        
        typename ImageSampleContainerType::ConstIterator fiter;
        typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
        typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();
        
        if( !this->m_SampleLastDimensionRandomly )
        {
            for( unsigned int i = 0; i < this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension); ++i )
            {
                this->m_RandomList[i]=i;
            }
        }
        else
        {
            this->SampleRandom( this->m_NumSamplesLastDimension, this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension), this->m_RandomList );
        }
        
        std::vector<RealType>               movingImageValue;
        
        for( fiter = fbegin; fiter != fend; ++fiter)
        {
            FixedImagePointType fixedPoint = ( *fiter ).Value().m_ImageCoordinates;
            FixedImageContinuousIndexType       voxelCoord;
            this->GetFixedImage()->TransformPhysicalPointToContinuousIndex( fixedPoint, voxelCoord );
            
            MovingImagePointType                mappedPointTemp;
            
            RealType                            movingImageValueTemp;
            movingImageValue.clear();
            movingImageValue.resize(this->m_NumSamplesLastDimension);
            
            
            bool sampleOk = true;
            
            for( unsigned int d = 0; d < this->m_NumSamplesLastDimension; d++ )
            {
                voxelCoord[ReducedFixedImageDimension] = this->m_RandomList[d];
                this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord, fixedPoint );
                sampleOk = sampleOk && this->TransformPoint( fixedPoint, mappedPointTemp);
                
                if ( sampleOk)
                {
                    sampleOk = sampleOk && this->IsInsideMovingMask( mappedPointTemp );
                }
                if ( sampleOk)
                {
                    movingImageValueTemp = 0;
                    sampleOk = sampleOk && this->EvaluateMovingImageValueAndDerivative(mappedPointTemp, movingImageValueTemp, 0);
                }
                if ( sampleOk)
                {
                    movingImageValue[d] = movingImageValueTemp;
                }
                
            }
            
            if ( sampleOk)
            {
                this->m_NumberOfPixelsCounted++;
                
                for( unsigned int d = 0; d < this->m_NumSamplesLastDimension; d++ )
                {
                    for( unsigned int e = 0; e < this->m_NumSamplesLastDimension; e++ )
                    {
                        if ( e != d )
                        {
                            this->UpdateJointPDFAndDerivatives(movingImageValue[d], movingImageValue[e], 0, 0, 0, 0, this->m_JointPDFs[this->m_RandomList[d]].GetPointer(), d);
                            
                        }
                    }
                }
            }
            
        }
        
        this->CheckNumberOfSamples( sampleContainer->Size(), this->m_NumberOfPixelsCounted );
        
        this->m_Alpha = 1.0 / static_cast< double >(this->m_NumberOfPixelsCounted);



    } // end ComputePDFs
    
    /**
     * ******************* ThreadedComputePDFs *******************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::ThreadedComputePDFs( ThreadIdType threadId )
    {
        const double nrOfSamplesPerThreads = static_cast< double >( this->m_NumSamplesLastDimension ) / static_cast< double >( this->m_NumberOfThreads )  ;
        
        unsigned long pos_begin = round(nrOfSamplesPerThreads * threadId);
        unsigned long pos_end   = round(nrOfSamplesPerThreads * ( threadId + 1 ));
        
        ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
        
        typename ImageSampleContainerType::ConstIterator fiter;
        typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
        typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();
        
        unsigned long numberOfPixelsCounted = 0;
        
        std::vector<RealType>               movingImageValue;
        
        for( fiter = fbegin; fiter != fend; ++fiter)
        {
            FixedImagePointType fixedPoint = ( *fiter ).Value().m_ImageCoordinates;
            FixedImageContinuousIndexType       voxelCoord;
            this->GetFixedImage()->TransformPhysicalPointToContinuousIndex( fixedPoint, voxelCoord );
            
            MovingImagePointType                mappedPointTemp;
            
            RealType                            movingImageValueTemp;
            movingImageValue.clear();
            movingImageValue.resize(this->m_NumSamplesLastDimension);
            
            bool sampleOk = true;
            for( unsigned int d = 0; d < this->m_NumSamplesLastDimension; d++ )
            {
                voxelCoord[ReducedFixedImageDimension] = this->m_RandomList[d];
                this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord, fixedPoint );
                sampleOk = sampleOk && this->TransformPoint( fixedPoint, mappedPointTemp);
                
                if ( sampleOk)
                {
                    sampleOk = sampleOk && this->IsInsideMovingMask( mappedPointTemp);
                }
                if ( sampleOk)
                {
                    movingImageValueTemp = 0;
                    sampleOk = sampleOk && this->EvaluateMovingImageValueAndDerivative(mappedPointTemp, movingImageValueTemp, 0);
                }
                if ( sampleOk)
                {
                    movingImageValue[d] = movingImageValueTemp;
                }
                
            }
            
            if ( sampleOk)
            {
                numberOfPixelsCounted++;
                
                for( unsigned int d = pos_begin; d < pos_end; d++ )
                {
                    for( unsigned int e = 0; e < this->m_NumSamplesLastDimension; e++ )
                    {
                        if ( e != d )
                        {
                            this->UpdateJointPDFAndDerivatives(movingImageValue[d], movingImageValue[e], 0, 0, 0, 0, this->m_JointPDFs[this->m_RandomList[d]].GetPointer(), d);
                            
                        }
                    }
                }
            }
            
        }
        
        this->m_ProbabilityAverageMutualInformationMetricGetValueAndDerivativePerThreadVariables[ threadId ].st_NumberOfPixelsCounted = numberOfPixelsCounted;

    }
    
    /**
     * ************************ ComputePDFsAndPDFDerivatives **************************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::ComputePDFsAndPDFDerivatives( const ParametersType & parameters ) const
    {
        if( !this->m_UseMultiThread )
        {
            return this->ComputePDFsAndPDFDerivativesSingleThreaded( parameters );
        }

        this->BeforeThreadedGetValueAndDerivative( parameters );
        
        this->InitializeThreadingParameters();
        
        this->LaunchComputePDFsAndPDFDerivativesThreaderCallback();
        
        this->AfterThreadedComputePDFsAndPDFDerivatives();
        
    }
    
    /**
     * **************** ComputePDFsAndPDFDerivativesThreaderCallback *******
     */
    
    template< class TFixedImage, class TMovingImage >
    ITK_THREAD_RETURN_TYPE
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::ComputePDFsAndPDFDerivativesThreaderCallback( void * arg )
    {
        ThreadInfoType * infoStruct = static_cast< ThreadInfoType * >( arg );
        ThreadIdType     threadId   = infoStruct->ThreadID;
        
        ProbabilityAverageMutualInformationMetricMultiThreaderParameterType * temp
        = static_cast< ProbabilityAverageMutualInformationMetricMultiThreaderParameterType * >( infoStruct->UserData );
        
        temp->m_Metric->ThreadedComputePDFsAndPDFDerivatives( threadId );
        
        return ITK_THREAD_RETURN_VALUE;
        
    }
    
    /**
     * *********************** LaunchComputePDFsAndPDFDerivativesThreaderCallback***************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::LaunchComputePDFsAndPDFDerivativesThreaderCallback( void ) const
    {
        typename ThreaderType::Pointer local_threader = ThreaderType::New();
        local_threader->SetNumberOfThreads( this->m_NumberOfThreads );
        local_threader->SetSingleMethod( this->ComputePDFsAndPDFDerivativesThreaderCallback, const_cast< void * >( static_cast< const void * >( &this->m_ProbabilityAverageMutualInformationMetricThreaderParameters ) ) );
        
        local_threader->SingleMethodExecute();
        
    }
    
    /**
     * ************************ ComputePDFsAndPDFDerivativesSingleThreaded *******************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::ComputePDFsAndPDFDerivativesSingleThreaded( const ParametersType & parameters ) const
    {
        for(unsigned int n = 0; n < this->m_NumSamplesLastDimension; n++)
        {
            this->m_JointPDFs[this->m_RandomList[n]]->FillBuffer( 0.0 );
            this->m_JointPDFDerivatives[this->m_RandomList[n]]->FillBuffer(0.0);
        }
        
        this->m_Alpha = 0.0;
        this->m_NumberOfPixelsCounted = 0;
        
        typedef typename DerivativeType::ValueType        DerivativeValueType;
        typedef typename TransformJacobianType::ValueType TransformJacobianValueType;
        
        this->BeforeThreadedGetValueAndDerivative( parameters );
        
        ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
        
        typename ImageSampleContainerType::ConstIterator fiter;
        typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
        typename ImageSampleContainerType::ConstIterator fend   = sampleContainer->End();
        
        if( !this->m_SampleLastDimensionRandomly )
        {
            for( unsigned int i = 0; i < this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension); ++i )
            {
                this->m_RandomList[i]=i;
            }
        }
        else
        {
            this->SampleRandom( this->m_NumSamplesLastDimension, this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension), this->m_RandomList );
        }
        
        std::vector<RealType>                   movingImageValue;
        std::vector<NonZeroJacobianIndicesType> nzji;
        std::vector<DerivativeType>             imageJacobian;
        
        for( fiter = fbegin; fiter != fend; ++fiter)
        {
            FixedImagePointType fixedPoint = ( *fiter ).Value().m_ImageCoordinates;
            FixedImageContinuousIndexType       voxelCoord;
            this->GetFixedImage()->TransformPhysicalPointToContinuousIndex( fixedPoint, voxelCoord );
            
            MovingImagePointType                    mappedPointTemp;
            
            MovingImageDerivativeType               movingImageDerivativeTemp;
            
            RealType                                movingImageValueTemp;
            movingImageValue.clear();
            movingImageValue.resize(this->m_NumSamplesLastDimension);
            
            TransformJacobianType                   jacobianTemp;
            
            NonZeroJacobianIndicesType              nzjiTemp( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices());
            nzji.clear();
            nzji.resize(this->m_NumSamplesLastDimension);
            
            DerivativeType                          imageJacobianTemp( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
            imageJacobian.clear();
            imageJacobian.resize(this->m_NumSamplesLastDimension);
            
            bool sampleOk = true;
            
            for( unsigned int d = 0; d < this->m_NumSamplesLastDimension; d++ )
            {
                voxelCoord[ReducedFixedImageDimension] = this->m_RandomList[d];
                this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord, fixedPoint );
                sampleOk = sampleOk && this->TransformPoint( fixedPoint, mappedPointTemp);
                
                if ( sampleOk)
                {
                    sampleOk = sampleOk && this->IsInsideMovingMask( mappedPointTemp );
                }
                if ( sampleOk)
                {
                    movingImageValueTemp = 0;
                    movingImageDerivativeTemp = 0;
                    sampleOk = sampleOk && this->EvaluateMovingImageValueAndDerivative(mappedPointTemp, movingImageValueTemp, &movingImageDerivativeTemp);
                }
                if ( sampleOk)
                {
                    sampleOk = sampleOk && this->EvaluateTransformJacobian( fixedPoint, jacobianTemp, nzjiTemp);
                }
                if (sampleOk)
                {
                    this->EvaluateTransformJacobianInnerProduct(jacobianTemp, movingImageDerivativeTemp, imageJacobianTemp );
                    movingImageValue[d] = movingImageValueTemp;
                    imageJacobian[d] = imageJacobianTemp;
                    nzji[d] = nzjiTemp;
                }
                
            }
            
            if ( sampleOk)
            {
                this->m_NumberOfPixelsCounted++;
                
                for( unsigned int d = 0; d < this->m_NumSamplesLastDimension; d++ )
                {
                    for( unsigned int e = 0; e < this->m_NumSamplesLastDimension; e++ )
                    {
                        if ( e != d )
                        {
                            this->UpdateJointPDFAndDerivatives(movingImageValue[d], movingImageValue[e], &(imageJacobian[d]), &(nzji[d]), &(imageJacobian[e]), &(nzji[e]), this->m_JointPDFs[this->m_RandomList[d]].GetPointer(), d);
                            
                        }
                    }
                }
            }
            
        }
        
        this->CheckNumberOfSamples( sampleContainer->Size(), this->m_NumberOfPixelsCounted );
        
        this->m_Alpha = 1.0 / static_cast< double >(this->m_NumberOfPixelsCounted);
        
    }

    
    /**
     * ******************* ThreadedComputePDFsAndPDFDerivatives *******************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::ThreadedComputePDFsAndPDFDerivatives( ThreadIdType threadId )
    {
        const double nrOfSamplesPerThreads = static_cast< double >( this->m_NumSamplesLastDimension ) / static_cast< double >( this->m_NumberOfThreads )  ;
        
        unsigned long pos_begin = round(nrOfSamplesPerThreads * threadId);
        unsigned long pos_end   = round(nrOfSamplesPerThreads * ( threadId + 1 ));
        
        ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
        
        typename ImageSampleContainerType::ConstIterator fiter;
        typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
        typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();
        
        unsigned long numberOfPixelsCounted = 0;
        
        std::vector<RealType>                   movingImageValue;
        std::vector<NonZeroJacobianIndicesType> nzji;
        std::vector<DerivativeType>             imageJacobian;
        
        for( fiter = fbegin; fiter != fend; ++fiter)
        {
            FixedImagePointType fixedPoint = ( *fiter ).Value().m_ImageCoordinates;
            FixedImageContinuousIndexType       voxelCoord;
            this->GetFixedImage()->TransformPhysicalPointToContinuousIndex( fixedPoint, voxelCoord );
            
            MovingImagePointType                    mappedPointTemp;
            
            MovingImageDerivativeType               movingImageDerivativeTemp;
            
            RealType                                movingImageValueTemp;
            movingImageValue.clear();
            movingImageValue.resize(this->m_NumSamplesLastDimension);
            double                                averageImageValue = 0.0;
            
            TransformJacobianType                   jacobianTemp;
            
            NonZeroJacobianIndicesType              nzjiTemp( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices());
            nzji.clear();
            nzji.resize(this->m_NumSamplesLastDimension);
            
            DerivativeType                          imageJacobianTemp( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
            imageJacobian.clear();
            imageJacobian.resize(this->m_NumSamplesLastDimension);
            
            bool sampleOk = true;
            
            for( unsigned int d = 0; d < this->m_NumSamplesLastDimension; d++ )
            {
                voxelCoord[ReducedFixedImageDimension] = this->m_RandomList[d];
                this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord, fixedPoint );
                sampleOk = sampleOk && this->TransformPoint( fixedPoint, mappedPointTemp);
                
                if ( sampleOk)
                {
                    sampleOk = sampleOk && this->IsInsideMovingMask( mappedPointTemp );
                }
                if ( sampleOk)
                {
                    movingImageValueTemp = 0;
                    movingImageDerivativeTemp = 0;
                    sampleOk = sampleOk && this->EvaluateMovingImageValueAndDerivative(mappedPointTemp, movingImageValueTemp, &movingImageDerivativeTemp);
                }
                if ( sampleOk)
                {
                    sampleOk = sampleOk && this->EvaluateTransformJacobian( fixedPoint, jacobianTemp, nzjiTemp);
                }
                if (sampleOk)
                {
                    this->EvaluateTransformJacobianInnerProduct(jacobianTemp, movingImageDerivativeTemp, imageJacobianTemp );
                    movingImageValue[d] = movingImageValueTemp;
                    imageJacobian[d] = imageJacobianTemp;
                    nzji[d] = nzjiTemp;
                    averageImageValue = averageImageValue + movingImageValueTemp;
                }
            }
            
            if ( sampleOk)
            {
                numberOfPixelsCounted++;
                
                averageImageValue = averageImageValue / this->m_NumSamplesLastDimension;
                
                for( unsigned int d = pos_begin; d < pos_end; d++ )
                {
                    for( unsigned int e = 0; e < this->m_NumSamplesLastDimension; e++ )
                    {
                        if ( e != d )
                        {
                            this->UpdateJointPDFAndDerivatives(movingImageValue[d], movingImageValue[e], &(imageJacobian[d]), &(nzji[d]), &(imageJacobian[e]), &(nzji[e]), this->m_JointPDFs[this->m_RandomList[d]].GetPointer(), d);
                            
                        }
                    }
                }
            }
        }
        
        this->m_ProbabilityAverageMutualInformationMetricGetValueAndDerivativePerThreadVariables[ threadId ].st_NumberOfPixelsCounted = numberOfPixelsCounted;
        
    }

    /**
     * ******************* AfterThreadedComputePDFsAndPDFDerivatives *******************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::AfterThreadedComputePDFsAndPDFDerivatives( void ) const
    {
        this->m_NumberOfPixelsCounted = this->m_ProbabilityAverageMutualInformationMetricGetValueAndDerivativePerThreadVariables[ 0 ].st_NumberOfPixelsCounted;
        
        ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
        this->CheckNumberOfSamples( sampleContainer->Size(), this->m_NumberOfPixelsCounted );
        
        this->m_Alpha = 1.0 / static_cast< double >(this->m_NumberOfPixelsCounted);

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
        typedef typename DerivativeType::ValueType        DerivativeValueType;
        typedef typename TransformJacobianType::ValueType TransformJacobianValueType;
        
        this->m_NumberOfPixelsCounted = 0;
        
        ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
        
        typename ImageSampleContainerType::ConstIterator fiter;
        typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
        typename ImageSampleContainerType::ConstIterator fend   = sampleContainer->End();
        
        std::vector<RealType>                   movingImageValue;
        std::vector<NonZeroJacobianIndicesType> nzji;
        std::vector<DerivativeType>             imageJacobian;
        
        for( fiter = fbegin; fiter != fend; ++fiter)
        {
            FixedImagePointType fixedPoint = ( *fiter ).Value().m_ImageCoordinates;
            FixedImageContinuousIndexType       voxelCoord;
            this->GetFixedImage()->TransformPhysicalPointToContinuousIndex( fixedPoint, voxelCoord );
            
            MovingImagePointType                    mappedPointTemp;
            
            MovingImageDerivativeType               movingImageDerivativeTemp;
            
            RealType                                movingImageValueTemp;
            movingImageValue.clear();
            movingImageValue.resize(this->m_NumSamplesLastDimension);
            
            
            TransformJacobianType                   jacobianTemp;
            
            NonZeroJacobianIndicesType              nzjiTemp( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices());
            nzji.clear();
            nzji.resize(this->m_NumSamplesLastDimension);
            
            DerivativeType                          imageJacobianTemp( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
            imageJacobian.clear();
            imageJacobian.resize(this->m_NumSamplesLastDimension);
            
            bool sampleOk = true;
            
            for( unsigned int d = 0; d < this->m_NumSamplesLastDimension; d++ )
            {
                voxelCoord[ReducedFixedImageDimension] = this->m_RandomList[d];
                this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord, fixedPoint );
                sampleOk = sampleOk && this->TransformPoint( fixedPoint, mappedPointTemp);
                
                if ( sampleOk)
                {
                    sampleOk = sampleOk && this->IsInsideMovingMask( mappedPointTemp );
                }
                if ( sampleOk)
                {
                    movingImageValueTemp = 0;
                    movingImageDerivativeTemp = 0;
                    sampleOk = sampleOk && this->EvaluateMovingImageValueAndDerivative(mappedPointTemp, movingImageValueTemp, &movingImageDerivativeTemp);
                }
                if ( sampleOk)
                {
                    sampleOk = sampleOk && this->EvaluateTransformJacobian( fixedPoint, jacobianTemp, nzjiTemp);
                }
                if (sampleOk)
                {
                    this->EvaluateTransformJacobianInnerProduct(jacobianTemp, movingImageDerivativeTemp, imageJacobianTemp );
                    movingImageValue[d] = movingImageValueTemp;
                    imageJacobian[d] = imageJacobianTemp;
                    nzji[d] = nzjiTemp;
                }
            }
            
            if ( sampleOk)
            {
                for( unsigned int d = 0; d < this->m_NumSamplesLastDimension; d++ )
                {
                    for( unsigned int e = 0; e < this->m_NumSamplesLastDimension; e++ )
                    {
                        if ( e != d )
                        {

                            this->UpdateDerivativeLowMemory(movingImageValue[d], movingImageValue[e], &(imageJacobian[d]), &(nzji[d]), &(imageJacobian[e]), &(nzji[e]), this->m_JointPDFs[this->m_RandomList[d]].GetPointer(),derivative, d);
                        }
                    }
                }
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
        const double nrOfSamplesPerThreads = static_cast< double >( this->m_NumSamplesLastDimension ) / static_cast< double >( this->m_NumberOfThreads )  ;
        
        unsigned long pos_begin = round(nrOfSamplesPerThreads * threadId);
        unsigned long pos_end   = round(nrOfSamplesPerThreads * ( threadId + 1 ));
        
        DerivativeType & derivative = this->m_GetValueAndDerivativePerThreadVariables[ threadId ].st_Derivative;
        
        derivative.Fill(NumericTraits< double >::Zero);
        
        ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
        
        typename ImageSampleContainerType::ConstIterator fiter;
        typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
        typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();
        
        std::vector<RealType>                   movingImageValue;
        std::vector<NonZeroJacobianIndicesType> nzji;
        std::vector<DerivativeType>             imageJacobian;
        
        for( fiter = fbegin; fiter != fend; ++fiter)
        {
            FixedImagePointType fixedPoint = ( *fiter ).Value().m_ImageCoordinates;
            FixedImageContinuousIndexType       voxelCoord;
            this->GetFixedImage()->TransformPhysicalPointToContinuousIndex( fixedPoint, voxelCoord );
            
            MovingImagePointType                    mappedPointTemp;
            
            MovingImageDerivativeType               movingImageDerivativeTemp;
            
            RealType                                movingImageValueTemp;
            movingImageValue.clear();
            movingImageValue.resize(this->m_NumSamplesLastDimension);
            
            TransformJacobianType                   jacobianTemp;
            
            NonZeroJacobianIndicesType              nzjiTemp( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices());
            nzji.clear();
            nzji.resize(this->m_NumSamplesLastDimension);
            
            DerivativeType                          imageJacobianTemp( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
            imageJacobian.clear();
            imageJacobian.resize(this->m_NumSamplesLastDimension);
            
            bool sampleOk = true;
            
            for( unsigned int d = 0; d < this->m_NumSamplesLastDimension; d++ )
            {
                voxelCoord[ReducedFixedImageDimension] = this->m_RandomList[d];
                this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint( voxelCoord, fixedPoint );
                sampleOk = sampleOk && this->TransformPoint( fixedPoint, mappedPointTemp);
                
                if ( sampleOk)
                {
                    sampleOk = sampleOk && this->IsInsideMovingMask( mappedPointTemp );
                }
                if ( sampleOk)
                {
                    movingImageValueTemp = 0;
                    movingImageDerivativeTemp = 0;
                    sampleOk = sampleOk && this->EvaluateMovingImageValueAndDerivative(mappedPointTemp, movingImageValueTemp, &movingImageDerivativeTemp);
                }
                if ( sampleOk)
                {
                    sampleOk = sampleOk && this->EvaluateTransformJacobian( fixedPoint, jacobianTemp, nzjiTemp);
                }
                if (sampleOk)
                {
                    this->EvaluateTransformJacobianInnerProduct(jacobianTemp, movingImageDerivativeTemp, imageJacobianTemp );
                    movingImageValue[d] = movingImageValueTemp;
                    imageJacobian[d] = imageJacobianTemp;
                    nzji[d] = nzjiTemp;
                }
                
            }
            
            if ( sampleOk)
            {
                for( unsigned int d = 0; d < this->m_NumSamplesLastDimension; d++ )
                {

                    for( unsigned int e = 0; e < this->m_NumSamplesLastDimension; e++ )
                    {
                        if ( e != d )
                        {
                            this->UpdateDerivativeLowMemory(movingImageValue[d], movingImageValue[e], &(imageJacobian[d]), &(nzji[d]), &(imageJacobian[e]), &(nzji[e]), this->m_JointPDFs[this->m_RandomList[d]].GetPointer(),derivative, d);
                        }
                    }
                }
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
        derivative = this->m_GetValueAndDerivativePerThreadVariables[ 0 ].st_Derivative;
        for( ThreadIdType i = 1; i < this->m_NumberOfThreads; ++i )
        {
            derivative += this->m_GetValueAndDerivativePerThreadVariables[ i ].st_Derivative;
        }

    }
    
    /**
     * ********************** UpdateJointPDFAndDerivatives ***************
     */
    template< class TFixedImage, class TMovingImage >
    void
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::UpdateJointPDFAndDerivatives(const RealType & fixedImageValue, const RealType & movingImageValue, const DerivativeType * imageJacobianFix, const NonZeroJacobianIndicesType * nzjiFix, const DerivativeType * imageJacobianMov, const NonZeroJacobianIndicesType * nzjiMov, JointPDFType * jointPDF, const unsigned int n ) const
    {
        typedef ImageScanlineIterator< JointPDFType > PDFIteratorType;
        
        /** Determine Parzen window arguments (see eq. 6 of Mattes paper [2]). */
        const double fixedImageParzenWindowTerm = fixedImageValue / this->m_FixedImageBinSizes[this->m_RandomList[n]] - this->m_FixedImageNormalizedMins[this->m_RandomList[n]];
        const double movingImageParzenWindowTerm = movingImageValue / this->m_MovingImageBinSizes[this->m_RandomList[n]] - this->m_MovingImageNormalizedMins[this->m_RandomList[n]];
        
        /** The lowest bin numbers affected by this pixel: */
        const OffsetValueType fixedImageParzenWindowIndex = static_cast< OffsetValueType >( vcl_floor(fixedImageParzenWindowTerm + this->m_FixedParzenTermToIndexOffset ) );
        const OffsetValueType movingImageParzenWindowIndex = static_cast< OffsetValueType >( vcl_floor(movingImageParzenWindowTerm + this->m_MovingParzenTermToIndexOffset ) );
        
        /** The Parzen values. */
        ParzenValueContainerType fixedParzenValues( this->m_JointPDFWindow.GetSize()[ 1 ] );
        ParzenValueContainerType movingParzenValues( this->m_JointPDFWindow.GetSize()[ 0 ] );
        this->EvaluateParzenValues(fixedImageParzenWindowTerm, fixedImageParzenWindowIndex, this->m_FixedKernel, fixedParzenValues );
        this->EvaluateParzenValues( movingImageParzenWindowTerm, movingImageParzenWindowIndex, this->m_MovingKernel, movingParzenValues );
        
        /** Position the JointPDFWindow. */
        JointPDFIndexType pdfWindowIndex;
        pdfWindowIndex[ 0 ] = movingImageParzenWindowIndex;
        pdfWindowIndex[ 1 ] = fixedImageParzenWindowIndex;
        
        if ((movingImageParzenWindowIndex > (this->m_NumberOfMovingHistogramBins - this->m_MovingKernelBSplineOrder -1)) || (fixedImageParzenWindowIndex > (this->m_NumberOfFixedHistogramBins - this->m_FixedKernelBSplineOrder -1 )) || (movingImageParzenWindowIndex < 0) || (fixedImageParzenWindowIndex < 0 ))
        {
            std::cout << " Warning: Index Out of Bounds:" << fixedImageValue << "\t" << fixedImageParzenWindowTerm << "\t" << this->m_FixedImageBinSizes[this->m_RandomList[n]] << "\t"<< this->m_FixedImageNormalizedMins[this->m_RandomList[n]] << "\t" << movingImageValue << "\t"  << movingImageParzenWindowTerm << "\t" << this->m_MovingImageBinSizes[this->m_RandomList[n]]  << "\t" << this->m_MovingImageNormalizedMins[this->m_RandomList[n]] << "\t" << n << "\t" << this->m_RandomList[n] << "\n";
        }
        else
        {
            /** For thread-safety, make a local copy of the support region,
             * and use that one. Because each thread will modify it.
             */
            JointPDFRegionType jointPDFWindow = this->m_JointPDFWindow;
            jointPDFWindow.SetIndex( pdfWindowIndex );
            PDFIteratorType it( jointPDF, jointPDFWindow );
        
            if( !imageJacobianFix )
            {
                /** Loop over the Parzen window region and increment the values. */
                for( unsigned int f = 0; f < fixedParzenValues.GetSize(); ++f )
                {
                    const double fv = fixedParzenValues[ f ];
                    for( unsigned int m = 0; m < movingParzenValues.GetSize(); ++m )
                    {
                        it.Value() += static_cast< PDFValueType >( fv * movingParzenValues[ m ] / static_cast<double>(this->m_NumSamplesLastDimension-1));
                        ++it;
                    }
                    it.NextLine();
                }
            }
            else
            {
                /** Compute the derivatives of the moving Parzen window. */
                ParzenValueContainerType derivativeMovingParzenValues(this->m_JointPDFWindow.GetSize()[ 0 ] );
                ParzenValueContainerType derivativeFixedParzenValues(this->m_JointPDFWindow.GetSize()[ 1 ] );
                
                this->EvaluateParzenValues( movingImageParzenWindowTerm, movingImageParzenWindowIndex, this->m_DerivativeMovingKernel, derivativeMovingParzenValues );
                this->EvaluateParzenValues( fixedImageParzenWindowTerm, fixedImageParzenWindowIndex, this->m_DerivativeMovingKernel, derivativeFixedParzenValues );
                
                const double etm = static_cast< double >( this->m_MovingImageBinSizes[this->m_RandomList[n]] );
                const double etf = static_cast< double >( this->m_FixedImageBinSizes[this->m_RandomList[n]] );

                for( unsigned int f = 0; f < fixedParzenValues.GetSize(); ++f )
                {
                    const double fv    = fixedParzenValues[f];
                    const double fv_etm = fv / etm;
                    for( unsigned int m = 0; m < movingParzenValues.GetSize(); ++m )
                    {
                        const double mv = movingParzenValues[m];
                        const double mv_etf = mv / etf;
                        it.Value() += static_cast< PDFValueType >( fv * mv / static_cast<double>(this->m_NumSamplesLastDimension-1));
                        
                        this->UpdateJointPDFDerivatives(it.GetIndex(), fv_etm * derivativeMovingParzenValues[ m ], mv_etf * derivativeFixedParzenValues[f], imageJacobianFix, nzjiFix, imageJacobianMov, nzjiMov, n );
                        ++it;
                    }
                    it.NextLine();
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
    ::UpdateJointPDFDerivatives( const JointPDFIndexType & pdfIndex, double factor, double factorr, const DerivativeType * imageJacobianFix, const NonZeroJacobianIndicesType * nzjiFix, const DerivativeType * imageJacobianMov, const NonZeroJacobianIndicesType * nzjiMov, const unsigned int n) const
    {
        PDFDerivativeValueType * derivPtr = this->m_JointPDFDerivatives[this->m_RandomList[n]]->GetBufferPointer() + ( pdfIndex[ 0 ] * this->m_JointPDFDerivatives[this->m_RandomList[n]]->GetOffsetTable()[ 1 ] ) + ( pdfIndex[ 1 ] * this->m_JointPDFDerivatives[this->m_RandomList[n]]->GetOffsetTable()[ 2 ] );
        typename DerivativeType::const_iterator imjac = (*imageJacobianFix).begin();
        
        for ( unsigned int i = 0; i < (*nzjiFix).size(); i++)
        {
            const unsigned int       mu  = (*nzjiFix)[i];
            PDFDerivativeValueType * ptr = derivPtr + mu;
            
            *( ptr ) -= static_cast< PDFDerivativeValueType >( (*imjac) * (factorr / static_cast<double>(this->m_NumSamplesLastDimension - 1) ) );
            imjac++;

        }

        imjac = (*imageJacobianMov).begin();
        
        for ( unsigned int i = 0; i < (*nzjiMov).size(); i++)
        {
            const unsigned int       mu  = (*nzjiMov)[i];
            PDFDerivativeValueType * ptr = derivPtr + mu;

            *( ptr ) -= static_cast< PDFDerivativeValueType >( (*imjac) * (factor / static_cast<double>(this->m_NumSamplesLastDimension - 1) ) );
            imjac++;
            
        }

    }
    
    /**
     * ******************* UpdateDerivativeLowMemory *******************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    ProbabilityAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::UpdateDerivativeLowMemory(const RealType & fixedImageValue, const RealType & movingImageValue, const DerivativeType * imageJacobianFix, const NonZeroJacobianIndicesType * nzjiFix, const DerivativeType * imageJacobianMov, const NonZeroJacobianIndicesType * nzjiMov, JointPDFType * jointPDF, DerivativeType & derivative, const unsigned int n ) const
    {
        const double fixedImageParzenWindowTerm = fixedImageValue / this->m_FixedImageBinSizes[this->m_RandomList[n]] - this->m_FixedImageNormalizedMins[this->m_RandomList[n]];
        const double movingImageParzenWindowTerm = movingImageValue / this->m_MovingImageBinSizes[this->m_RandomList[n]] - this->m_MovingImageNormalizedMins[this->m_RandomList[n]];
        
        const int fixedImageParzenWindowIndex = static_cast< int >( vcl_floor(fixedImageParzenWindowTerm + this->m_FixedParzenTermToIndexOffset ) );
        const int movingImageParzenWindowIndex = static_cast< int >( vcl_floor(movingImageParzenWindowTerm + this->m_MovingParzenTermToIndexOffset ) );
        
        if ((movingImageParzenWindowIndex > (this->m_NumberOfMovingHistogramBins - this->m_MovingKernelBSplineOrder - 1)) || (fixedImageParzenWindowIndex > (this->m_NumberOfFixedHistogramBins - this->m_FixedKernelBSplineOrder - 1 )) || (movingImageParzenWindowIndex < 0 )|| (fixedImageParzenWindowIndex < 0))
        {
            std::cout << fixedImageValue << "\t" << movingImageValue << "\t" << this->m_MovingImageMinLimits[this->m_RandomList[n]] << "\t"<< movingImageParzenWindowIndex << "\t" << (movingImageParzenWindowIndex > (this->m_NumberOfMovingHistogramBins - this->m_MovingKernelBSplineOrder - 1)) << n << "\n";
        }
        else
        {
            ParzenValueContainerType fixedParzenValues( this->m_JointPDFWindow.GetSize()[ 1 ] );
            ParzenValueContainerType movingParzenValues( this->m_JointPDFWindow.GetSize()[ 0 ] );
            this->EvaluateParzenValues(fixedImageParzenWindowTerm, fixedImageParzenWindowIndex, this->m_FixedKernel, fixedParzenValues );
            this->EvaluateParzenValues(movingImageParzenWindowTerm, movingImageParzenWindowIndex, this->m_MovingKernel, movingParzenValues );
            
            ParzenValueContainerType derivativeMovingParzenValues(this->m_JointPDFWindow.GetSize()[ 0 ] );
            ParzenValueContainerType derivativeFixedParzenValues(this->m_JointPDFWindow.GetSize()[ 1 ] );
            this->EvaluateParzenValues(movingImageParzenWindowTerm, movingImageParzenWindowIndex,this->m_DerivativeMovingKernel, derivativeMovingParzenValues );
            // Moet eigen fixed zijn, maar wordt niet aangemaakt in avergemetric
            this->EvaluateParzenValues( fixedImageParzenWindowTerm, fixedImageParzenWindowIndex, this->m_DerivativeMovingKernel, derivativeFixedParzenValues );
            
            const double etm = static_cast< double >( this->m_MovingImageBinSizes[this->m_RandomList[n]] );
            const double etf = static_cast< double >( this->m_FixedImageBinSizes[this->m_RandomList[n]]);
            
            double regSum = 0.0;
            double extraSum = 0.0;

            for( unsigned int f = 0; f < fixedParzenValues.GetSize(); ++f )
            {
                const double fv    = fixedParzenValues[f];
                const double fv_etm = fv / etm;
                for( unsigned int m = 0; m < movingParzenValues.GetSize(); ++m )
                {
                    const double mv = movingParzenValues[m];
                    const double mv_etf = mv / etf;
                    
                    extraSum = extraSum + (*(this->m_PRatioArray[this->m_RandomList[n]]))[ f + fixedImageParzenWindowIndex ][ m + movingImageParzenWindowIndex ] * mv_etf * derivativeFixedParzenValues[ f ] ;

                    regSum = regSum + (*(this->m_PRatioArray[this->m_RandomList[n]]))[ f + fixedImageParzenWindowIndex ][ m + movingImageParzenWindowIndex ] * fv_etm * derivativeMovingParzenValues[ m ];

                }
            }
            
            typename DerivativeType::const_iterator imjac = (*imageJacobianFix).begin();

            for ( unsigned int i = 0; i < (*nzjiFix).size(); i++)
            {
                const unsigned int       mu  = (*nzjiFix)[i];
                
                derivative[ mu ] += static_cast< PDFDerivativeValueType >( (*imjac) * (extraSum / static_cast<double>(this->m_NumSamplesLastDimension - 1) ) );
                imjac++;
                
            }
            
            imjac = (*imageJacobianMov).begin();
            
            for ( unsigned int i = 0; i < (*nzjiMov).size(); i++)
            {
                const unsigned int       mu  = (*nzjiMov)[i];
                
                derivative[ mu ] += static_cast< PDFDerivativeValueType >( (*imjac) * (regSum / static_cast<double>(this->m_NumSamplesLastDimension - 1) ) );
                imjac++;
                
            }


        }
    }
}

#endif // end #ifndef _itkProbabilityAverageMutualInformationMetric_HXX__
