#ifndef _itkPAverageMutualInformationMetric_HXX__
#define _itkPAverageMutualInformationMetric_HXX__

#include "itkPAverageMutualInformationMetric.h"

namespace itk
{
    /**
     * ********************* Constructor ****************************
     */
    
    template< class TFixedImage, class TMovingImage >
    PAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::PAverageMutualInformationMetric()
    {
        m_SubtractMean = false;
        m_TransformIsStackTransform = true;
        m_UseDerivative = true;
        m_UseExplicitPDFDerivatives = true;
        
        this->SetUseImageSampler( true );
        this->SetUseFixedImageLimiter( true );
        this->SetUseMovingImageLimiter( true );
        
    }


    /**
     * ******************* Destructor *******************************
     */
    
    template< class TFixedImage, class TMovingImage >
    PAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::~PAverageMutualInformationMetric()
    {
        
    }
    
    /**
     * ********************* Initialize *****************************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    PAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::Initialize( void ) throw ( ExceptionObject )
    {

        this->Superclass::Initialize();
        
        if(!this->m_SampleLastDimensionRandomly)
        {
            m_NumSamplesLastDimension = this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension);
        }
        
        this->InitializeVectors();
        
        this->InitializeKernels();
                
    }
    
    /**
     * ********************* InitializeVectors *****************************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    PAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::InitializeVectors( void )
    {
        this->m_FixedImageTrueMins.assign(this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension), 0.0);
        this->m_FixedImageMinLimits.assign(this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension), 0.0);
        this->m_FixedImageTrueMaxs.assign(this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension), 1.0);
        this->m_FixedImageMaxLimits.assign(this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension), 1.0);
        this->m_FixedImageBinSizes.assign(this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension), 0.0);
        this->m_FixedImageNormalizedMins.assign(this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension), 0.0);
        
        this->m_MovingImageTrueMins.assign(this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension), 0.0);
        this->m_MovingImageMinLimits.assign(this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension), 0.0);
        this->m_MovingImageTrueMaxs.assign(this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension), 1.0);
        this->m_MovingImageMaxLimits.assign(this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension), 1.0);
        this->m_MovingImageBinSizes.assign(this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension), 0.0);
        this->m_MovingImageNormalizedMins.assign(this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension), 0.0);
        
        this->m_FixedImageMarginalPDFs.resize(this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension));
        this->m_MovingImageMarginalPDFs.resize(this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension));
        this->m_JointPDFs.resize(this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension));
        this->m_JointPDFDerivatives.resize(this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension));
        
        this->m_Values.assign(this->GetFixedImage()->GetLargestPossibleRegion().GetSize(ReducedFixedImageDimension),0.0);
        this->m_RandomList.assign(this->m_NumSamplesLastDimension, 0.0);
        
        if(!this->m_SampleLastDimensionRandomly)
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
        
    }

    /**
     * ********************* InitializeKernels *****************************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    PAverageMutualInformationMetric< TFixedImage, TMovingImage >
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
     * ******************* SampleRandom *******************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    PAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::SampleRandom( const int n, const int m, std::vector< unsigned int > & numbers ) const
    {
        numbers.clear();
        
        Statistics::MersenneTwisterRandomVariateGenerator::Pointer randomGenerator = Statistics::MersenneTwisterRandomVariateGenerator::GetInstance();
        
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
    }
    
    /**
     * *********************** NormalizeJointPDF ***********************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    PAverageMutualInformationMetric< TFixedImage, TMovingImage >
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
    PAverageMutualInformationMetric< TFixedImage, TMovingImage >
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
     * ********************** EvaluateParzenValues ***************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    PAverageMutualInformationMetric< TFixedImage, TMovingImage >
    ::EvaluateParzenValues( double parzenWindowTerm, OffsetValueType parzenWindowIndex, const KernelFunctionType * kernel, ParzenValueContainerType & parzenValues ) const
    {
        const unsigned int max_i = parzenValues.GetSize();
        for( unsigned int i = 0; i < max_i; ++i, ++parzenWindowIndex )
        {
            parzenValues[ i ] = kernel->Evaluate(static_cast< double >( parzenWindowIndex ) - parzenWindowTerm );
        }
        
    }
    
    /**
     * ******************* ComputeValueandPRatioArray *******************
     */
    
    template < class TFixedImage, class TMovingImage >
    void
    PAverageMutualInformationMetric<TFixedImage,TMovingImage>
    ::ComputeValueAndPRatioArray( double & MI, unsigned int n ) const
    {
        this->m_PRatioArray[this->m_RandomList[n]]->Fill(0.0);
        
        typedef ImageScanlineConstIterator< JointPDFType > JointPDFIteratorType;
        typedef typename MarginalPDFType::const_iterator   MarginalPDFIteratorType;
        
        JointPDFIteratorType jointPDFit(this->m_JointPDFs[this->m_RandomList[n]], this->m_JointPDFs[this->m_RandomList[n]]->GetLargestPossibleRegion() );
        MarginalPDFIteratorType       fixedPDFit  = this->m_FixedImageMarginalPDFs[this->m_RandomList[n]]->begin();
        const MarginalPDFIteratorType fixedPDFend = this->m_FixedImageMarginalPDFs[this->m_RandomList[n]]->end();
        MarginalPDFIteratorType       movingPDFit;
        const MarginalPDFIteratorType movingPDFbegin = this->m_MovingImageMarginalPDFs[this->m_RandomList[n]]->begin();
        const MarginalPDFIteratorType movingPDFend   = this->m_MovingImageMarginalPDFs[this->m_RandomList[n]]->end();
        
        PDFValueType sum         = 0.0;
        unsigned int fixedIndex  = 0;
        unsigned int movingIndex = 0;
        while( fixedPDFit != fixedPDFend )
        {
            const double fixedPDFValue = *fixedPDFit;
            
            movingPDFit = movingPDFbegin;
            movingIndex = 0;
            
            while( movingPDFit != movingPDFend )
            {
                const PDFValueType movingPDFValue = *movingPDFit;
                const PDFValueType jointPDFValue  = jointPDFit.Value();
                const PDFValueType movPDFfixPDF = movingPDFValue * fixedPDFValue;
                
                if( jointPDFValue > 1e-16 && movPDFfixPDF > 1e-16 )
                {
                    const PDFValueType pRatio = vcl_log( jointPDFValue / movPDFfixPDF );
                    
                    (*(this->m_PRatioArray[this->m_RandomList[n]]))[ fixedIndex ][ movingIndex ] = static_cast< PRatioType >(this->m_Alpha*pRatio);
                    
                    sum += jointPDFValue * ( pRatio);
                }
                
                ++movingPDFit;
                ++jointPDFit;
                ++movingIndex;
                
            }
            
            ++fixedPDFit;
            jointPDFit.NextLine();
            ++fixedIndex;
        }
        
        MI = sum;
    }
}

#endif // end #ifndef _itkPAverageMutualInformationMetric_HXX__
