#ifndef _itkLinearGroupwiseNJE_HXX__
#define _itkLinearGroupwiseNJE_HXX__

#include "itkLinearGroupwiseNJE.h"
#include "itkImageScanlineIterator.h"

#ifdef ELASTIX_USE_OPENMP
#include <omp.h>
#endif

namespace itk
{
    /**
     * ********************* Constructor ******************************
     */
    
    template< class TFixedImage, class TMovingImage >
    LinearGroupwiseNJE< TFixedImage, TMovingImage >
    ::LinearGroupwiseNJE()
    {
        this->m_UseDerivative = true;
        this->SetUseImageSampler( true );
        
        this->m_LinearGroupwiseNJEThreaderParameters.m_Metric = this;
        
        this->m_LinearGroupwiseNJEGetValueAndDerivativePerThreadVariables     = NULL;
        this->m_LinearGroupwiseNJEGetValueAndDerivativePerThreadVariablesSize = 0;

    }
    
    /**
     * ******************* Destructor *******************
     */
    
    template< class TFixedImage, class TMovingImage >
    LinearGroupwiseNJE< TFixedImage, TMovingImage >
    ::~LinearGroupwiseNJE()
    {
        delete[] this->m_LinearGroupwiseNJEGetValueAndDerivativePerThreadVariables;

    }
    
    /**
     * ******************* Initialize *******************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearGroupwiseNJE< TFixedImage, TMovingImage >
    ::Initialize( void ) throw ( ExceptionObject )
    {
        this->Superclass::Initialize();
        
        if(!this->m_SampleLastDimensionRandomly)
        {
            this->m_NumSamplesLastDimension = this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension);
        }

        this->InitializeKernels();

        this->InitializeVectors();
        
        this->InitializeHistograms();

    }
    
    /**
     * ********************* InitializeKernels *****************************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearGroupwiseNJE< TFixedImage, TMovingImage >
    ::InitializeKernels( void )
    {
        switch( this->m_FixedKernelBSplineOrder )
        {
            case 0:
                this->m_FixedKernel = BSplineKernelFunction< 0 >::New(); break;
            case 1:
                this->m_FixedKernel = BSplineKernelFunction< 1 >::New(); break;
            case 2:
                this->m_FixedKernel = BSplineKernelFunction< 2 >::New(); break;
            case 3:
                this->m_FixedKernel = BSplineKernelFunction< 3 >::New(); break;
            default:
                itkExceptionMacro( << "The following FixedKernelBSplineOrder is not implemented: " << this->m_FixedKernelBSplineOrder );
        }
        
        switch( this->m_MovingKernelBSplineOrder )
        {
            case 0:
                this->m_MovingKernel = BSplineKernelFunction< 0 >::New();
                this->m_DerivativeMovingKernel = BSplineDerivativeKernelFunction< 1 >::New();
                break;
            case 1:
                this->m_MovingKernel           = BSplineKernelFunction< 1 >::New();
                this->m_DerivativeMovingKernel = BSplineDerivativeKernelFunction< 1 >::New();
                break;
            case 2:
                this->m_MovingKernel           = BSplineKernelFunction< 2 >::New();
                this->m_DerivativeMovingKernel = BSplineDerivativeKernelFunction< 2 >::New();
                break;
            case 3:
                this->m_MovingKernel           = BSplineKernelFunction< 3 >::New();
                this->m_DerivativeMovingKernel = BSplineDerivativeKernelFunction< 3 >::New();
                break;
            default:
                itkExceptionMacro( << "The following MovingKernelBSplineOrder is not implemented: " << this->m_MovingKernelBSplineOrder );
        }
        
        JointPDFSizeType parzenWindowSize;
        parzenWindowSize[ 0 ] = this->m_MovingKernelBSplineOrder + 1;
        parzenWindowSize[ 1 ] = this->m_FixedKernelBSplineOrder + 1;
        this->m_JointPDFWindow.SetSize( parzenWindowSize );
        this->m_JointPDFWindow.SetSize( parzenWindowSize );
        
        this->m_FixedParzenTermToIndexOffset = 0.5 - static_cast< double >( this->m_FixedKernelBSplineOrder ) / 2.0;
        this->m_MovingParzenTermToIndexOffset = 0.5 - static_cast< double >( this->m_MovingKernelBSplineOrder ) / 2.0;
        
    }
    
    /**
     * ********************* InitializeVectors ******************************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearGroupwiseNJE< TFixedImage, TMovingImage >
    ::InitializeVectors(void)
    {
        const unsigned int lastDim     = this->GetFixedImage()->GetImageDimension() - 1;
        const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );

        this->m_FixedImageTrueMins.assign(lastDimSize, 0.0);
        this->m_FixedImageMinLimits.assign(lastDimSize, 0.0);
        this->m_FixedImageTrueMaxs.assign(lastDimSize, 1.0);
        this->m_FixedImageMaxLimits.assign(lastDimSize, 1.0);
        this->m_FixedImageBinSizes.assign(lastDimSize, 0.0);
        this->m_FixedImageNormalizedMins.assign(lastDimSize, 0.0);
        
        this->m_FixedImageMarginalPDFs.resize(lastDimSize);
        this->m_MovingImageMarginalPDFs.resize(lastDimSize);
        this->m_JointPDFs.resize(lastDimSize);
        this->m_JointPDFDerivatives.resize(lastDimSize);

        std::vector<double> intensityConstants(lastDimSize,0.0);
        for(unsigned int d = 0; d < lastDimSize; d++)
        {
            MovingImagePixelType trueMinTemp = NumericTraits< MovingImagePixelType >::max();
            MovingImagePixelType trueMaxTemp = NumericTraits< MovingImagePixelType >::NonpositiveMin();
            
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
                
                region.SetIndex(ReducedFixedImageDimension, d);
                region.SetSize(ReducedFixedImageDimension, 1);
                
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
            
            if( this->m_FixedImageTrueMins[d] < 0)
            {
                intensityConstants[d] = 0 - this->m_FixedImageTrueMins[d];
            }
            
        }
        
        this->m_TemplateImage->SetIntensityConstants(intensityConstants);
        
        this->m_MovingImageTrueMin = static_cast<MovingImagePixelType>(this->m_TemplateImage->DetermineHistogramMin(this->m_FixedImageTrueMins, lastDimSize));
        this->m_MovingImageTrueMax = static_cast<MovingImagePixelType>(this->m_TemplateImage->DetermineHistogramMax(this->m_FixedImageTrueMaxs, lastDimSize));
        
        this->m_MovingImageMinLimit = static_cast< MovingImageLimiterOutputType >( this->m_MovingImageTrueMin - this->m_MovingImageLimitRangeRatio * (this->m_MovingImageTrueMax - this->m_MovingImageTrueMin));
        this->m_MovingImageMaxLimit = static_cast< MovingImageLimiterOutputType >(this->m_MovingImageTrueMax + this->m_MovingImageLimitRangeRatio * (this->m_MovingImageTrueMax - this->m_MovingImageTrueMin));
        
        for(unsigned int d = 0; d < lastDimSize; d++)
        {
            this->m_FixedImageMarginalPDFs[d] = new MarginalPDFType;
            this->m_FixedImageMarginalPDFs[d]->SetSize(this->m_NumberOfFixedHistogramBins);
            this->m_MovingImageMarginalPDFs[d] = new MarginalPDFType;
            this->m_MovingImageMarginalPDFs[d]->SetSize( this->m_NumberOfMovingHistogramBins );
            this->m_JointPDFs[d] = JointPDFType::New();
        }
        
        this->m_RandomList.assign(this->m_NumSamplesLastDimension, 0.0);
        
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

    }
    
    /**
     * ********************* InitializeHistograms ******************************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearGroupwiseNJE< TFixedImage, TMovingImage >
    ::InitializeHistograms(void)
    {
        const unsigned int lastDim     = this->GetFixedImage()->GetImageDimension() - 1;
        const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );
        
        int fixedPadding  = this->m_FixedKernelBSplineOrder / 2;
        int movingPadding = this->m_MovingKernelBSplineOrder / 2;
        
        const double smallNumberRatio = 0.001;
        
        const double smallNumberMoving = smallNumberRatio * ( this->m_MovingImageMaxLimit - this->m_MovingImageMinLimit ) / static_cast< double >( this->m_NumberOfMovingHistogramBins - 2 * movingPadding - 1 );
        
        const double fixedHistogramWidth = static_cast< double >( static_cast< OffsetValueType >( this->m_NumberOfFixedHistogramBins ) - 2.0 * fixedPadding - 1.0 );
        const double movingHistogramWidth = static_cast< double >( static_cast< OffsetValueType >( this->m_NumberOfMovingHistogramBins ) - 2.0 * movingPadding - 1.0 );
        
        this->m_MovingImageBinSize = ( this->m_MovingImageMaxLimit - this->m_MovingImageMinLimit + 2.0 * smallNumberMoving ) / movingHistogramWidth;
        
        this->m_MovingImageBinSize = vnl_math_max( this->m_MovingImageBinSize, 1e-10 );
        this->m_MovingImageBinSize = vnl_math_min( this->m_MovingImageBinSize, 1e+10 );
        this->m_MovingImageNormalizedMin = ( this->m_MovingImageMinLimit - smallNumberMoving ) / this->m_MovingImageBinSize - static_cast< double >( movingPadding );
        
        for(unsigned int d = 0; d < lastDimSize; d++)
        {
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
            this->m_PRatioArray.resize(this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension));
            
            for(unsigned int d = 0; d < lastDimSize; d++)
            {
                this->m_PRatioArray[d] = new PRatioArrayType;
                this->m_PRatioArray[d]->SetSize( this->GetNumberOfFixedHistogramBins(), this->GetNumberOfMovingHistogramBins() );
                this->m_PRatioArray[d]->Fill( itk::NumericTraits< PRatioType >::Zero  );
            }
        }
        
    }
    
    /**
     * *********************** NormalizeJointPDF ***********************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearGroupwiseNJE< TFixedImage, TMovingImage >
    ::NormalizeJointPDF( JointPDFType * pdf, const double & factor ) const
    {
        
        typedef ImageScanlineIterator< JointPDFType > JointPDFIteratorType;
        JointPDFIteratorType it( pdf, pdf->GetBufferedRegion() );
        const PDFValueType   castfac = static_cast< PDFValueType >( factor );
        while( !it.IsAtEnd() )
        {
            while( !it.IsAtEndOfLine() )
            {
                it.Value() *= castfac;
                ++it;
            }
            it.NextLine();
        }
        
    }
    
    /**
     * ************************ ComputeMarginalPDF ***********************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearGroupwiseNJE< TFixedImage, TMovingImage >
    ::ComputeMarginalPDF( const JointPDFType * jointPDF , MarginalPDFType* marginalPDF, const unsigned int & direction ) const
    {
        typedef ImageLinearConstIteratorWithIndex< JointPDFType > JointPDFLinearIterator;
        JointPDFLinearIterator linearIter( jointPDF, jointPDF->GetBufferedRegion() );
        linearIter.SetDirection( direction );
        linearIter.GoToBegin();
        unsigned int marginalIndex = 0;
        while( !linearIter.IsAtEnd() )
        {
            PDFValueType sum = 0.0;
            while( !linearIter.IsAtEndOfLine() )
            {
                sum += linearIter.Get();
                ++linearIter;
            }
            (*marginalPDF)[ marginalIndex ] = sum;
            linearIter.NextLine();
            ++marginalIndex;
        }
        
    }
    
    /**
     * ********************** ComputeLogMarginalPDF***********************
     */
    
    template< class TFixedImage, class TMovingImage  >
    void
    LinearGroupwiseNJE< TFixedImage, TMovingImage >
    ::ComputeLogMarginalPDF( MarginalPDFType* pdf ) const
    {
        typedef typename MarginalPDFType::iterator MarginalPDFIteratorType;
        
        MarginalPDFIteratorType       PDFit  = pdf->begin();
        const MarginalPDFIteratorType PDFend = pdf->end();
        
        while( PDFit != PDFend )
        {
            if( ( *PDFit ) > 1e-16 )
            {
                ( *PDFit ) = vcl_log( *PDFit );
            }
            else
            {
                ( *PDFit ) = 0.0;
            }
            ++PDFit;
        }
        
    }   // end ComputeLogMarginalPDF

    
    /**
     * ********************** EvaluateParzenValues ***************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearGroupwiseNJE< TFixedImage, TMovingImage >
    ::EvaluateParzenValues( double parzenWindowTerm, OffsetValueType parzenWindowIndex, const KernelFunctionType * kernel, ParzenValueContainerType & parzenValues ) const
    {
        const unsigned int max_i = parzenValues.GetSize();
        for( unsigned int i = 0; i < max_i; ++i, ++parzenWindowIndex )
        {
            parzenValues[ i ] = kernel->Evaluate(static_cast< double >( parzenWindowIndex ) - parzenWindowTerm );
        }
        
    }
    
    /**
     * ******************* GetValue *******************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearGroupwiseNJE< TFixedImage, TMovingImage >
    ::GetValue(const ParametersType & parameters, std::vector<MeasureType> & values, bool & minimize ) const
    {
        const unsigned int lastDim     = this->GetFixedImage()->GetImageDimension() - 1;
        const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );

        values.assign(this->m_NumSamplesLastDimension,0.0);

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
        
        this->ComputePDFs( parameters );
        
        for(unsigned int s = 0; s < this->m_NumSamplesLastDimension; s++)
        {
            this->NormalizeJointPDF( this->m_JointPDFs[this->m_RandomList[s]], 1.0 / static_cast<double>(this->m_NumberOfPixelsCountedVector[this->m_RandomList[s]]) );
            this->ComputeMarginalPDF( this->m_JointPDFs[this->m_RandomList[s]], this->m_FixedImageMarginalPDFs[this->m_RandomList[s]], 0 );
            this->ComputeMarginalPDF( this->m_JointPDFs[this->m_RandomList[s]], this->m_MovingImageMarginalPDFs[this->m_RandomList[s]], 1 );
            this->ComputeLogMarginalPDF( this->m_FixedImageMarginalPDFs[this->m_RandomList[s]] );
            this->ComputeLogMarginalPDF( this->m_MovingImageMarginalPDFs[this->m_RandomList[s]] );
        }
        
        typedef ImageLinearConstIteratorWithIndex< JointPDFType >           JointPDFIteratorType;
        typedef ImageLinearConstIteratorWithIndex<JointPDFDerivativesType >JointPDFDerivativesIteratorType;
        typedef typename MarginalPDFType::const_iterator                    MarginalPDFIteratorType;
        
        for(unsigned int s = 0; s < this->m_NumSamplesLastDimension; s++)
        {
            JointPDFIteratorType jointPDFit(this->m_JointPDFs[this->m_RandomList[s]], this->m_JointPDFs[this->m_RandomList[s]]->GetLargestPossibleRegion() );
            jointPDFit.SetDirection( 0 );
            jointPDFit.GoToBegin();
            
            MarginalPDFIteratorType       fixedPDFit   = this->m_FixedImageMarginalPDFs[this->m_RandomList[s]]->begin();
            const MarginalPDFIteratorType fixedPDFend  = this->m_FixedImageMarginalPDFs[this->m_RandomList[s]]->end();
            MarginalPDFIteratorType       movingPDFit  = this->m_MovingImageMarginalPDFs[this->m_RandomList[s]]->begin();
            const MarginalPDFIteratorType movingPDFend = this->m_MovingImageMarginalPDFs[this->m_RandomList[s]]->end();
            
            double sumnum = 0.0;
            double sumden = 0.0;
            while( fixedPDFit != fixedPDFend )
            {
                const double logFixedImagePDFValue = *fixedPDFit;
                movingPDFit = this->m_MovingImageMarginalPDFs[this->m_RandomList[s]]->begin();
                
                while( movingPDFit != movingPDFend )
                {
                    const double jointPDFValue       = jointPDFit.Get();

                    sumnum -= jointPDFValue * ( logFixedImagePDFValue );

                    if( jointPDFValue > 1e-16 )
                    {
                        sumden -= jointPDFValue * vcl_log( jointPDFValue );
                    }
                    
                    ++movingPDFit;
                    ++jointPDFit;
                    
                }
                ++fixedPDFit;
                jointPDFit.NextLine();
                
            }
            values[s] = static_cast< MeasureType >( sumnum / sumden );
        }
        minimize = false;
    }
        
    /**
     * ******************* GetValueAndDerivative *******************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearGroupwiseNJE< TFixedImage, TMovingImage >
    ::GetValueAndDerivative( const ParametersType & parameters, std::vector<MeasureType> & values, std::vector<DerivativeType> & derivatives, bool & minimize ) const
    {
        if ( this->GetUseExplicitPDFDerivatives() )
        {
            this->GetValueAndAnalyticalDerivative(parameters, values, derivatives);
        }
        else
        {
            this->GetValueAndAnalyticalDerivativeLowMemory(parameters, values, derivatives );
        }
        minimize = false;
    }
    
    /**
     * *********************  GetValueAndAnalyticalDerivative ****************************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearGroupwiseNJE< TFixedImage, TMovingImage >
    ::GetValueAndAnalyticalDerivative( const ParametersType & parameters, std::vector<MeasureType> & values, std::vector<DerivativeType> & derivatives ) const
    {
        const unsigned int lastDim     = this->GetFixedImage()->GetImageDimension() - 1;
        const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );
        
        std::vector<double> entropies;
        
        values.assign(this->m_NumSamplesLastDimension,0.0);
        entropies.assign(this->m_NumSamplesLastDimension,0.0);
        
        derivatives.resize(lastDimSize);
        for(unsigned int d= 0; d < lastDimSize; d++)
        {
            derivatives[d] = DerivativeType( this->GetNumberOfParameters() );
            derivatives[d].Fill(NumericTraits< DerivativeValueType >::Zero);
        }

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
        
        this->ComputePDFsAndPDFDerivatives( parameters);
        
        for(unsigned int s = 0; s < this->m_NumSamplesLastDimension; s++)
        {
            this->NormalizeJointPDF( this->m_JointPDFs[this->m_RandomList[s]], 1.0 / static_cast<double>(this->m_NumberOfPixelsCountedVector[this->m_RandomList[s]]) );
            this->ComputeMarginalPDF( this->m_JointPDFs[this->m_RandomList[s]], this->m_FixedImageMarginalPDFs[this->m_RandomList[s]], 0 );
            this->ComputeMarginalPDF( this->m_JointPDFs[this->m_RandomList[s]], this->m_MovingImageMarginalPDFs[this->m_RandomList[s]], 1 );
            this->ComputeLogMarginalPDF( this->m_FixedImageMarginalPDFs[this->m_RandomList[s]] );
            this->ComputeLogMarginalPDF( this->m_MovingImageMarginalPDFs[this->m_RandomList[s]] );

        }
        
        typedef ImageLinearConstIteratorWithIndex< JointPDFType > JointPDFIteratorType;
        typedef ImageLinearConstIteratorWithIndex<JointPDFDerivativesType >                       JointPDFDerivativesIteratorType;
        typedef typename MarginalPDFType::const_iterator          MarginalPDFIteratorType;
        
        typedef typename DerivativeType::iterator        DerivativeIteratorType;
        typedef typename DerivativeType::const_iterator  DerivativeConstIteratorType;
        
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
            
            double sumnum = 0.0;
            double sumden = 0.0;
            
            while( fixedPDFit != fixedPDFend )
            {
                const double logFixedImagePDFValue = *fixedPDFit;
                movingPDFit = this->m_MovingImageMarginalPDFs[this->m_RandomList[s]]->begin();
                while( movingPDFit != movingPDFend )
                {
                    const double logMovingImagePDFValue = *movingPDFit;
                    const double jointPDFValue       = jointPDFit.Get();
                    
                    sumnum -= jointPDFValue * ( logFixedImagePDFValue );
                    
                    if( jointPDFValue > 1e-16 )
                    {
                        sumden -= jointPDFValue * vcl_log( jointPDFValue );
                    }
                    
                    ++movingPDFit;
                    ++jointPDFit;
                    
                }
                
                ++fixedPDFit;
                jointPDFit.NextLine();
                
            }
            entropies[s] = sumden;
            values[s] = static_cast< MeasureType >( sumnum / sumden );
        }
        
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
            
            DerivativeIteratorType        derivit      = derivatives[s].begin();
            const DerivativeIteratorType  derivbegin   = derivatives[s].begin();
            const DerivativeIteratorType  derivend     = derivatives[s].end();
            
            while( fixedPDFit != fixedPDFend )
            {
                const double logFixedImagePDFValue = *fixedPDFit;
                movingPDFit = this->m_MovingImageMarginalPDFs[this->m_RandomList[s]]->begin();
                while( movingPDFit != movingPDFend )
                {
                    const double logMovingImagePDFValue = *movingPDFit;
                    const double jointPDFValue       = jointPDFit.Get();
                    
                    if( jointPDFValue > 1e-16)
                    {
                        derivit = derivbegin;
                        const double pRatio      = (values[s] * vcl_log(jointPDFValue) - logFixedImagePDFValue) / entropies[s];
                        const double pRatioAlpha = pRatio / static_cast<double>(this->m_NumberOfPixelsCountedVector[this->m_RandomList[s]]);

                        while( derivit != derivend )
                        {
                            ( *derivit ) += jointPDFDerivativesit.Get() * pRatioAlpha;
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
            
        }
        
    }
    
    /**
     * *********************  GetValueAndAnalyticalDerivativeLowMemory ***************************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearGroupwiseNJE< TFixedImage, TMovingImage >
    ::GetValueAndAnalyticalDerivativeLowMemory( const ParametersType & parameters, std::vector<MeasureType> & values, std::vector<DerivativeType> & derivatives ) const
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

        this->ComputePDFs( parameters );
        double MI =0;
        
        for(unsigned int s = 0; s < this->m_NumSamplesLastDimension; s++)
        {
            this->NormalizeJointPDF( this->m_JointPDFs[this->m_RandomList[s]], 1.0 / static_cast<double>(this->m_NumberOfPixelsCountedVector[this->m_RandomList[s]]) );
            this->ComputeMarginalPDF( this->m_JointPDFs[this->m_RandomList[s]], this->m_FixedImageMarginalPDFs[this->m_RandomList[s]], 0 );
            this->ComputeMarginalPDF( this->m_JointPDFs[this->m_RandomList[s]], this->m_MovingImageMarginalPDFs[this->m_RandomList[s]], 1 );
            this->ComputeLogMarginalPDF( this->m_FixedImageMarginalPDFs[this->m_RandomList[s]] );
            this->ComputeLogMarginalPDF( this->m_MovingImageMarginalPDFs[this->m_RandomList[s]] );
            this->ComputeValueAndPRatioArray( MI, s);
            values[s] = MI;
        }

        this->ComputeDerivativeLowMemory( derivatives);

    }
        
    /**
     * ************************ ComputePDFs **************************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearGroupwiseNJE< TFixedImage, TMovingImage >
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
    LinearGroupwiseNJE< TFixedImage, TMovingImage >
    ::ComputePDFsThreaderCallback( void * arg )
    {
        ThreadInfoType * infoStruct = static_cast< ThreadInfoType * >( arg );
        ThreadIdType     threadId   = infoStruct->ThreadID;
        
        LinearGroupwiseNJEMultiThreaderParameterType * temp
        = static_cast< LinearGroupwiseNJEMultiThreaderParameterType * >( infoStruct->UserData );
        
        temp->m_Metric->ThreadedComputePDFs( threadId );
        
        return ITK_THREAD_RETURN_VALUE;
        
    }
    
    /**
     * *********************** LaunchComputePDFsThreaderCallback***************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearGroupwiseNJE< TFixedImage, TMovingImage >
    ::LaunchComputePDFsThreaderCallback( void ) const
    {
        typename ThreaderType::Pointer local_threader = ThreaderType::New();
        local_threader->SetNumberOfThreads( this->m_NumberOfThreads );
        local_threader->SetSingleMethod( this->ComputePDFsThreaderCallback, const_cast< void * >( static_cast< const void * >( &this->m_LinearGroupwiseNJEThreaderParameters ) ) );
        
        local_threader->SingleMethodExecute();
        
    }
    
    /**
     * ********************* InitializeThreadingParameters ****************************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearGroupwiseNJE< TFixedImage, TMovingImage >
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
        
        if( this->m_LinearGroupwiseNJEGetValueAndDerivativePerThreadVariablesSize != this->m_NumberOfThreads )
        {
            delete[] this->m_LinearGroupwiseNJEGetValueAndDerivativePerThreadVariables;
            this->m_LinearGroupwiseNJEGetValueAndDerivativePerThreadVariables = new AlignedLinearGroupwiseNJEGetValueAndDerivativePerThreadStruct[this->m_NumberOfThreads ];
            this->m_LinearGroupwiseNJEGetValueAndDerivativePerThreadVariablesSize = this->m_NumberOfThreads;
        }
        
        for( ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i )
        {
            this->m_LinearGroupwiseNJEGetValueAndDerivativePerThreadVariables[ i ].st_NumberOfPixelsCounted = NumericTraits< SizeValueType >::Zero;
            this->m_LinearGroupwiseNJEGetValueAndDerivativePerThreadVariables[ i ].st_NumberOfPixelsCountedVector.assign(lastDimSize, NumericTraits< SizeValueType >::Zero);
            this->m_LinearGroupwiseNJEGetValueAndDerivativePerThreadVariables[ i ].st_JointPDFs.resize(lastDimSize);
            this->m_LinearGroupwiseNJEGetValueAndDerivativePerThreadVariables[ i ].st_Derivatives.resize(lastDimSize);

            for(unsigned int d = 0; d < lastDimSize; d++)
            {
                JointPDFPointer & jointPDF = this->m_LinearGroupwiseNJEGetValueAndDerivativePerThreadVariables[ i ].st_JointPDFs[ d ];
                if( jointPDF.IsNull() ) { jointPDF = JointPDFType::New(); }
                if( jointPDF->GetLargestPossibleRegion() != jointPDFRegion )
                {
                    jointPDF->SetRegions( jointPDFRegion );
                    jointPDF->Allocate();
                }
                
                if (this->m_UseExplicitPDFDerivatives)
                {
                    this->m_LinearGroupwiseNJEGetValueAndDerivativePerThreadVariables[ i ].st_Derivatives.clear();
                }
                
            }
            
            
        }
        
     } // end InitializeThreadingParameters()
    
    /**
     * ******************* AfterThreadedComputePDFs *******************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearGroupwiseNJE< TFixedImage, TMovingImage >
    ::AfterThreadedComputePDFs( void ) const
    {
        const unsigned int lastDim     = this->GetFixedImage()->GetImageDimension() - 1;
        const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );

        this->m_NumberOfPixelsCountedVector.assign(lastDimSize,0.0);
        for( ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i )
        {
            this->m_NumberOfPixelsCounted += this->m_LinearGroupwiseNJEGetValueAndDerivativePerThreadVariables[ i ].st_NumberOfPixelsCounted;

            for(unsigned int s = 0; s < this->m_NumSamplesLastDimension; s++)
            {
                this->m_NumberOfPixelsCountedVector[this->m_RandomList[s]] += this->m_LinearGroupwiseNJEGetValueAndDerivativePerThreadVariables[ i ].st_NumberOfPixelsCountedVector[this->m_RandomList[s]];
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
                itT[ i ] = JointPDFIteratorType(this-> m_LinearGroupwiseNJEGetValueAndDerivativePerThreadVariables[ i ].st_JointPDFs[this->m_RandomList[s]], this->m_JointPDFs[this->m_RandomList[s]]->GetBufferedRegion() );
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
    LinearGroupwiseNJE< TFixedImage, TMovingImage >
    ::ComputePDFsSingleThreaded( const ParametersType & parameters ) const
    {
        const unsigned int lastDim     = this->GetFixedImage()->GetImageDimension() - 1;
        const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );

        for(unsigned int s = 0; s < this->m_NumSamplesLastDimension; s++)
        {
            this->m_JointPDFs[this->m_RandomList[s]]->FillBuffer( 0.0 );
        }

        this->m_NumberOfPixelsCountedVector.assign(lastDimSize,0.0);
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
                    this->UpdateJointPDFAndDerivatives(imageValues, 0, 0, this->m_JointPDFs[positions[o]].GetPointer(), o, positions);
                    
                }
            }
            
        }
        
        this->CheckNumberOfSamples( sampleContainer->Size(), this->m_NumberOfPixelsCounted );
    }
    
    /**
     * ******************* ThreadedComputePDFs *******************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearGroupwiseNJE< TFixedImage, TMovingImage >
    ::ThreadedComputePDFs( ThreadIdType threadId )
    {
        const unsigned int lastDim     = this->GetFixedImage()->GetImageDimension() - 1;
        const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );
        
        std::vector<JointPDFPointer> & jointPDFs = this->m_LinearGroupwiseNJEGetValueAndDerivativePerThreadVariables[ threadId ].st_JointPDFs;
        std::vector<SizeValueType> & NumberOfPixelsCountedVector = this->m_LinearGroupwiseNJEGetValueAndDerivativePerThreadVariables[ threadId ].st_NumberOfPixelsCountedVector;
        SizeValueType & NumberOfPixelsCounted = this->m_LinearGroupwiseNJEGetValueAndDerivativePerThreadVariables[ threadId ].st_NumberOfPixelsCounted;
        
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
                    else if(movingImageValueTemp < this->m_FixedImageTrueMins[this->m_RandomList[ s ]] )
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
                    this->UpdateJointPDFAndDerivatives(imageValues, 0, 0, jointPDFs[positions[o]].GetPointer(), o, positions);
                    
                }
            }
            
        }
        
    }
    
    /**
     * ************************ ComputePDFsAndPDFDerivatives **************************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearGroupwiseNJE< TFixedImage, TMovingImage >
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
    LinearGroupwiseNJE< TFixedImage, TMovingImage >
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
        
        this->m_NumberOfPixelsCountedVector.assign(lastDimSize,0.0);
        this->m_NumberOfPixelsCounted = 0;

        this->BeforeThreadedGetValueAndDerivative( parameters );
        
        ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
        
        typename ImageSampleContainerType::ConstIterator fiter;
        typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
        typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();
        
        for( fiter = fbegin; fiter != fend; ++fiter)
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
                    this->UpdateJointPDFAndDerivatives(imageValues, &(imageJacobians), &(nzjis), this->m_JointPDFs[positions[o]].GetPointer(), o, positions);
                    
                }
            }
            
        }
        
        this->CheckNumberOfSamples( sampleContainer->Size(), this->m_NumberOfPixelsCounted );
    }

    
    /**
     * ******************* ComputeValueandPRatioArray *******************
     */
    
    template < class TFixedImage, class TMovingImage >
    void
    LinearGroupwiseNJE<TFixedImage,TMovingImage>
    ::ComputeValueAndPRatioArray( double & nMI, unsigned int n ) const
    {
        this->m_PRatioArray[this->m_RandomList[n]]->Fill(0.0);
        
        typedef ImageScanlineConstIterator< JointPDFType > JointPDFIteratorType;
        
        typedef typename MarginalPDFType::const_iterator   MarginalPDFIteratorType;
        
        JointPDFIteratorType jointPDFit(this->m_JointPDFs[this->m_RandomList[n]], this->m_JointPDFs[this->m_RandomList[n]]->GetLargestPossibleRegion() );
        jointPDFit.GoToBegin();
        MarginalPDFIteratorType       fixedPDFit  = this->m_FixedImageMarginalPDFs[this->m_RandomList[n]]->begin();
        const MarginalPDFIteratorType fixedPDFend = this->m_FixedImageMarginalPDFs[this->m_RandomList[n]]->end();
        MarginalPDFIteratorType       movingPDFit;
        const MarginalPDFIteratorType movingPDFbegin = this->m_MovingImageMarginalPDFs[this->m_RandomList[n]]->begin();
        const MarginalPDFIteratorType movingPDFend   = this->m_MovingImageMarginalPDFs[this->m_RandomList[n]]->end();
        
        double sumnum = 0.0;
        double sumden = 0.0;
        while( fixedPDFit != fixedPDFend )
        {
            const double logFixedImagePDFValue = *fixedPDFit;
            
            movingPDFit = movingPDFbegin;
            
            while( movingPDFit != movingPDFend )
            {
                const PDFValueType jointPDFValue  = jointPDFit.Value();
                
                sumnum -= jointPDFValue * ( logFixedImagePDFValue);
                
                if( jointPDFValue > 1e-16 )
                {
                    sumden -= jointPDFValue * vcl_log( jointPDFValue );
                }
                
                ++movingPDFit;
                ++jointPDFit;
                
            }
            
            ++fixedPDFit;
            jointPDFit.NextLine();
        }
        
        nMI = static_cast< MeasureType >( sumnum / sumden );
        
        fixedPDFit  = this->m_FixedImageMarginalPDFs[this->m_RandomList[n]]->begin();
        jointPDFit.GoToBegin();
        unsigned int fixedIndex  = 0;
        unsigned int movingIndex = 0;
        while( fixedPDFit != fixedPDFend )
        {
            const double logFixedImagePDFValue = *fixedPDFit;
            
            movingPDFit = movingPDFbegin;
            movingIndex = 0;
            
            while( movingPDFit != movingPDFend )
            {
                const PDFValueType jointPDFValue  = jointPDFit.Value();
                
                if( jointPDFValue > 1e-16 )
                {
                    const PDFValueType pRatio = (nMI * vcl_log(jointPDFValue) - logFixedImagePDFValue) / sumden;
                    (*(this->m_PRatioArray[this->m_RandomList[n]]))[ fixedIndex ][ movingIndex ] = static_cast< PRatioType >(pRatio / this->m_NumberOfPixelsCountedVector[this->m_RandomList[n]]);
                    
                }
                
                ++movingPDFit;
                ++jointPDFit;
                ++movingIndex;
                
            }
            
            ++fixedPDFit;
            jointPDFit.NextLine();
            ++fixedIndex;
        }

    }

    /**
     * ******************** ComputeDerivativeLowMemory *******************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearGroupwiseNJE< TFixedImage, TMovingImage >
    ::ComputeDerivativeLowMemory( std::vector<DerivativeType> & derivatives) const
    {
        if( !this->m_UseMultiThread )
        {
            return this->ComputeDerivativeLowMemorySingleThreaded( derivatives);
        }
        
        this->LaunchComputeDerivativeLowMemoryThreaderCallback();
        
        this->AfterThreadedComputeDerivativeLowMemory( derivatives );
    }
    
    /**
     * **************** ComputeDerivativeLowMemoryThreaderCallback *******
     */
    
    template< class TFixedImage, class TMovingImage >
    ITK_THREAD_RETURN_TYPE
    LinearGroupwiseNJE< TFixedImage, TMovingImage >
    ::ComputeDerivativeLowMemoryThreaderCallback( void * arg )
    {
        ThreadInfoType * infoStruct = static_cast< ThreadInfoType * >( arg );
        ThreadIdType     threadId   = infoStruct->ThreadID;
        
        LinearGroupwiseNJEMultiThreaderParameterType * temp
        = static_cast< LinearGroupwiseNJEMultiThreaderParameterType * >( infoStruct->UserData );
        
        temp->m_Metric->ThreadedComputeDerivativeLowMemory( threadId );
        
        return ITK_THREAD_RETURN_VALUE;
        
    }
    
    
    /**
     * *********************** LaunchComputeDerivativeLowMemoryThreaderCallback***************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearGroupwiseNJE< TFixedImage, TMovingImage >
    ::LaunchComputeDerivativeLowMemoryThreaderCallback( void ) const
    {
        typename ThreaderType::Pointer local_threader = ThreaderType::New();
        local_threader->SetNumberOfThreads( this->m_NumberOfThreads );
        local_threader->SetSingleMethod( this->ComputeDerivativeLowMemoryThreaderCallback, const_cast< void * >( static_cast< const void * >( &this->m_LinearGroupwiseNJEThreaderParameters ) ) );
        
        local_threader->SingleMethodExecute();
        
    }
    
    /**
     * ******************* AfterThreadedComputeDerivativeLowMemory *******************
     */
    
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearGroupwiseNJE< TFixedImage, TMovingImage >
    ::AfterThreadedComputeDerivativeLowMemory( std::vector<DerivativeType> & derivatives ) const
    {
        for( ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i )
        {
            std::vector<DerivativeType> & derivative = this->m_LinearGroupwiseNJEGetValueAndDerivativePerThreadVariables[ i ].st_Derivatives;
            
            for(unsigned int s = 0; s < this->m_NumSamplesLastDimension; s++)
            {
                derivatives[this->m_RandomList[s]] += derivative[this->m_RandomList[s]];
            }
            
        }

    }

    
    /**
     * *********************** ComputeDerivativeLowMemorySingleThreaded***************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearGroupwiseNJE< TFixedImage, TMovingImage >
    ::ComputeDerivativeLowMemorySingleThreaded( std::vector<DerivativeType> & derivatives) const
    {
        const unsigned int lastDim     = this->GetFixedImage()->GetImageDimension() - 1;
        const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );
        
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

                for( unsigned int o = 0; o < numSamplesOk; o++ )
                {
                    this->UpdateDerivativeLowMemory(imageValues, &(imageJacobians), &(nzjis), o, positions, derivatives);
                    
                }
            }
        }
    }
    
    /**
     * *********************** ThreadedComputeDerivativeLowMemory***************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearGroupwiseNJE< TFixedImage, TMovingImage >
    ::ThreadedComputeDerivativeLowMemory( ThreadIdType threadId )
    {

        const unsigned int lastDim     = this->GetFixedImage()->GetImageDimension() - 1;
        const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );
        
        std::vector<DerivativeType> & derivatives = this->m_LinearGroupwiseNJEGetValueAndDerivativePerThreadVariables[ threadId ].st_Derivatives;
        
        derivatives.resize(lastDimSize);
        
        for(unsigned int s = 0; s < this->m_NumSamplesLastDimension; s++)
        {
            derivatives[this->m_RandomList[s]] = DerivativeType( this->GetNumberOfParameters() );
            derivatives[this->m_RandomList[s]].Fill(0.0);
        }

        std::vector<RealType>                   movingImageValue;
        std::vector<NonZeroJacobianIndicesType> nzji;
        std::vector<DerivativeType>             imageJacobian;
        
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
                    else if(movingImageValueTemp < this->m_FixedImageTrueMins[this->m_RandomList[ s ]] )
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

                for( unsigned int o = 0; o < numSamplesOk; o++ )
                {
                    this->UpdateDerivativeLowMemory(imageValues, &(imageJacobians), &(nzjis), o, positions, derivatives);
                    
                }
            }
        }
    }

    /**
     * ********************** UpdateJointPDFAndDerivatives ***************
     */
    template< class TFixedImage, class TMovingImage >
    void
    LinearGroupwiseNJE< TFixedImage, TMovingImage >
    ::UpdateJointPDFAndDerivatives(const std::vector<double> & fixedImageValue, const std::vector<DerivativeType> * imageJacobian, const std::vector<NonZeroJacobianIndicesType> * nzji, JointPDFType * jointPDF, const unsigned int n, const std::vector<unsigned int> & positions ) const
    {
        typedef ImageScanlineIterator< JointPDFType > PDFIteratorType;
        
        double movingImageValue = this->m_TemplateImage->CalculateIntensity(fixedImageValue, fixedImageValue.size(), positions, n);

        const double fixedImageParzenWindowTerm = fixedImageValue[n] / this->m_FixedImageBinSizes[positions[n]] - this->m_FixedImageNormalizedMins[positions[n]];
        const double movingImageParzenWindowTerm = movingImageValue / this->m_MovingImageBinSize - this->m_MovingImageNormalizedMin;
        const OffsetValueType fixedImageParzenWindowIndex = static_cast< OffsetValueType >( vcl_floor(fixedImageParzenWindowTerm + this->m_FixedParzenTermToIndexOffset ) );
        const OffsetValueType movingImageParzenWindowIndex = static_cast< OffsetValueType >( vcl_floor(movingImageParzenWindowTerm + this->m_MovingParzenTermToIndexOffset ) );

        ParzenValueContainerType fixedParzenValues( this->m_JointPDFWindow.GetSize()[ 1 ] );
        ParzenValueContainerType movingParzenValues( this->m_JointPDFWindow.GetSize()[ 0 ] );
        this->EvaluateParzenValues( fixedImageParzenWindowTerm, fixedImageParzenWindowIndex, this->m_FixedKernel, fixedParzenValues );
        this->EvaluateParzenValues(movingImageParzenWindowTerm, movingImageParzenWindowIndex, this->m_MovingKernel, movingParzenValues );
        
        JointPDFIndexType pdfWindowIndex;
        
        pdfWindowIndex[ 1 ] = fixedImageParzenWindowIndex;
        pdfWindowIndex[ 0 ] = movingImageParzenWindowIndex;
        
        JointPDFRegionType jointPDFWindow = this->m_JointPDFWindow;
        jointPDFWindow.SetIndex( pdfWindowIndex );
        PDFIteratorType it( jointPDF, jointPDFWindow );
            
        if( !imageJacobian )
        {
            for( unsigned int f = 0; f < fixedParzenValues.GetSize(); f++ )
            {
                const double fv = fixedParzenValues[ f ];
                for( unsigned int m = 0; m < movingParzenValues.GetSize(); m++ )
                {
                    it.Value() += static_cast< PDFValueType >( fv * movingParzenValues[ m ] );
                    ++it;
                }
                it.NextLine();
            }
        }
            
        else
        {
            ParzenValueContainerType derivativeMovingParzenValues(this->m_JointPDFWindow.GetSize()[ 0 ] );
            ParzenValueContainerType derivativeFixedParzenValues(this->m_JointPDFWindow.GetSize()[ 1 ] );
                
            this->EvaluateParzenValues( movingImageParzenWindowTerm, movingImageParzenWindowIndex, this->m_DerivativeMovingKernel, derivativeMovingParzenValues );
            this->EvaluateParzenValues( fixedImageParzenWindowTerm, fixedImageParzenWindowIndex, this->m_DerivativeMovingKernel, derivativeFixedParzenValues );
                
            const double etm = static_cast< double >( this->m_MovingImageBinSize );
            const double etf = static_cast< double >( this->m_FixedImageBinSizes[positions[n]] );
                
            for( unsigned int f = 0; f < fixedParzenValues.GetSize(); ++f )
            {
                const double fv    = fixedParzenValues[f];
                const double fv_etm = fv / etm;
                for( unsigned int m = 0; m < movingParzenValues.GetSize(); ++m )
                {
                    const double mv = movingParzenValues[m];
                    const double mv_etf = mv / etf;
                    it.Value() += static_cast< PDFValueType >( fv * mv );
                    UpdateJointPDFDerivatives(fixedImageValue, movingImageValue, it.GetIndex(),fv_etm * derivativeMovingParzenValues[ m ], mv_etf * derivativeFixedParzenValues[f], imageJacobian, nzji, n, positions);
                        
                    ++it;
                }
                it.NextLine();
            }
        }
    }

    
    /**
     * *************** UpdateJointPDFDerivatives ***************************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearGroupwiseNJE< TFixedImage, TMovingImage >
    ::UpdateJointPDFDerivatives( const std::vector<double> & fixedImageValue, const double & movingImageValue, const JointPDFIndexType & pdfIndex, double factor, double factorr, const std::vector<DerivativeType> * imageJacobian, const std::vector<NonZeroJacobianIndicesType> * nzji, const unsigned int n, const std::vector<unsigned int> & positions ) const
    {
        PDFDerivativeValueType * derivPtr = this->m_JointPDFDerivatives[positions[n]]->GetBufferPointer() + ( pdfIndex[ 0 ] * this->m_JointPDFDerivatives[positions[n]]->GetOffsetTable()[ 1 ] ) + ( pdfIndex[ 1 ] * this->m_JointPDFDerivatives[positions[n]]->GetOffsetTable()[ 2 ] );
        
        for ( unsigned int o = 0; o < positions.size(); o++)
        {
            typename DerivativeType::const_iterator imjac = (*imageJacobian)[o].begin();
            for ( unsigned int i = 0; i < (*nzji)[o].size(); i++)
            {
                const unsigned int       mu  = (*nzji)[o][i];
                PDFDerivativeValueType * ptr = derivPtr + mu;
                
                if ( o == n)
                {
                    *( ptr ) -= static_cast< PDFDerivativeValueType >( (*imjac) * (factor * this->m_TemplateImage->CalculateRatio(fixedImageValue[o],movingImageValue,positions[o],positions[n],positions.size()) + factorr) );
                }
                else
                {
                    *( ptr ) -= static_cast< PDFDerivativeValueType >( (*imjac) * factor * this->m_TemplateImage->CalculateRatio(fixedImageValue[o],movingImageValue,positions[o],positions[n],positions.size()));
                }
                imjac++;
                    
            }
            
        }
        
    }
    
    /**
     * ******************* UpdateDerivativeLowMemory *******************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    LinearGroupwiseNJE< TFixedImage, TMovingImage >
    ::UpdateDerivativeLowMemory(const std::vector<double> & fixedImageValue, const std::vector<DerivativeType> * imageJacobian, const std::vector<NonZeroJacobianIndicesType> * nzji, const unsigned int n, const std::vector<unsigned int> & positions, std::vector<DerivativeType> & derivatives) const
    {

        double movingImageValue = this->m_TemplateImage->CalculateIntensity(fixedImageValue, fixedImageValue.size(), positions, n);

        const double fixedImageParzenWindowTerm = fixedImageValue[n] / this->m_FixedImageBinSizes[positions[n]] - this->m_FixedImageNormalizedMins[positions[n]];
        const double movingImageParzenWindowTerm = movingImageValue / this->m_MovingImageBinSize - this->m_MovingImageNormalizedMin;
        const OffsetValueType fixedImageParzenWindowIndex = static_cast< OffsetValueType >( vcl_floor(fixedImageParzenWindowTerm + this->m_FixedParzenTermToIndexOffset ) );
        const OffsetValueType movingImageParzenWindowIndex = static_cast< OffsetValueType >( vcl_floor(movingImageParzenWindowTerm + this->m_MovingParzenTermToIndexOffset ) );
        
        ParzenValueContainerType fixedParzenValues( this->m_JointPDFWindow.GetSize()[ 1 ] );
        ParzenValueContainerType movingParzenValues( this->m_JointPDFWindow.GetSize()[ 0 ] );
        this->EvaluateParzenValues(fixedImageParzenWindowTerm, fixedImageParzenWindowIndex, this->m_FixedKernel, fixedParzenValues );
        this->EvaluateParzenValues(movingImageParzenWindowTerm, movingImageParzenWindowIndex, this->m_MovingKernel, movingParzenValues );

        ParzenValueContainerType derivativeMovingParzenValues(this->m_JointPDFWindow.GetSize()[ 0 ] );
        ParzenValueContainerType derivativeFixedParzenValues(this->m_JointPDFWindow.GetSize()[ 1 ] );
            
        this->EvaluateParzenValues( movingImageParzenWindowTerm, movingImageParzenWindowIndex, this->m_DerivativeMovingKernel, derivativeMovingParzenValues );
            //MOET EIGENLIJK FIXED ZIJN
        this->EvaluateParzenValues( fixedImageParzenWindowTerm, fixedImageParzenWindowIndex, this->m_DerivativeMovingKernel, derivativeFixedParzenValues );
            
        const double etm = static_cast< double >( this->m_MovingImageBinSize );
        const double etf = static_cast< double >( this->m_FixedImageBinSizes[positions[n]] );
            
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

        for ( unsigned int o = 0; o < positions.size(); o++)
        {
            typename DerivativeType::const_iterator imjac = (*imageJacobian)[o].begin();
            for ( unsigned int i = 0; i < (*nzji)[o].size(); i++)
            {
                const unsigned int mu  = (*nzji)[o][i];
                if ( o == n)
                {
                    derivatives[positions[o]][ mu ] -= static_cast<  double >( (*imjac) * (extraSum + regSum * this->m_TemplateImage->CalculateRatio(fixedImageValue[o],movingImageValue,positions[o],positions[n],positions.size()) ) );
                }
                else
                {
                    derivatives[positions[o]][ mu ] -= static_cast<  double >( (*imjac) * regSum * this->m_TemplateImage->CalculateRatio(fixedImageValue[o],movingImageValue,positions[o],positions[n],positions.size()) );
                }
                    
                imjac++;
            }
        }
    }
}

#endif // end #ifndef _itkLinearGroupwiseNJE_HXX__
