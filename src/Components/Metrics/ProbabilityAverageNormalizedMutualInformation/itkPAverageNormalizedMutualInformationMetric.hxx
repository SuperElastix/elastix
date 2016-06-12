#ifndef _itkPAverageNormalizedMutualInformationMetric_HXX__
#define _itkPAverageNormalizedMutualInformationMetric_HXX__

#include "itkPAverageNormalizedMutualInformationMetric.h"

namespace itk
{
    /**
     * ********************* Constructor ****************************
     */
    
    template< class TFixedImage, class TMovingImage >
    PAverageNormalizedMutualInformationMetric< TFixedImage, TMovingImage >
    ::PAverageNormalizedMutualInformationMetric()
    {
        m_SubtractMean = false;
        m_TransformIsStackTransform = true;
        m_UseDerivative = true;
        m_UseExplicitPDFDerivatives = true;
        
        this->SetUseImageSampler( true );
        
    }


    /**
     * ******************* Destructor *******************************
     */
    
    template< class TFixedImage, class TMovingImage >
    PAverageNormalizedMutualInformationMetric< TFixedImage, TMovingImage >
    ::~PAverageNormalizedMutualInformationMetric()
    {
        
    }
    
    /**
     * ********************* Initialize *****************************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    PAverageNormalizedMutualInformationMetric< TFixedImage, TMovingImage >
    ::Initialize( void ) throw ( ExceptionObject )
    {

        const unsigned int lastDim     = this->GetFixedImage()->GetImageDimension() - 1;
        const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );

        this->Superclass::Initialize();
        
        if(!this->m_SampleLastDimensionRandomly)
        {
            this->m_NumSamplesLastDimension = lastDimSize;
        }

        this->InitializeVectors();
        
        this->InitializeKernels();
                
    }
    
    /**
     * ********************* InitializeVectors *****************************
     */
    
    template< class TFixedImage, class TMovingImage >
    void
    PAverageNormalizedMutualInformationMetric< TFixedImage, TMovingImage >
    ::InitializeVectors( void )
    {
        const unsigned int lastDim     = this->GetFixedImage()->GetImageDimension() - 1;
        const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize( lastDim );

        this->m_FixedImageTrueMins.assign(lastDimSize, 0.0);
        this->m_FixedImageMinLimits.assign(lastDimSize, 0.0);
        this->m_FixedImageTrueMaxs.assign(lastDimSize, 1.0);
        this->m_FixedImageMaxLimits.assign(lastDimSize, 1.0);
        this->m_FixedImageBinSizes.assign(lastDimSize, 0.0);
        this->m_FixedImageNormalizedMins.assign(lastDimSize, 0.0);
        
        this->m_MovingImageTrueMins.assign(lastDimSize, 0.0);
        this->m_MovingImageMinLimits.assign(lastDimSize, 0.0);
        this->m_MovingImageTrueMaxs.assign(lastDimSize, 1.0);
        this->m_MovingImageMaxLimits.assign(lastDimSize, 1.0);
        this->m_MovingImageBinSizes.assign(lastDimSize, 0.0);
        this->m_MovingImageNormalizedMins.assign(lastDimSize, 0.0);
        
        this->m_FixedImageMarginalPDFs.resize(lastDimSize);
        this->m_MovingImageMarginalPDFs.resize(lastDimSize);
        this->m_JointPDFs.resize(lastDimSize);
        this->m_JointPDFDerivatives.resize(lastDimSize);
        
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
    PAverageNormalizedMutualInformationMetric< TFixedImage, TMovingImage >
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
    PAverageNormalizedMutualInformationMetric< TFixedImage, TMovingImage >
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
    PAverageNormalizedMutualInformationMetric< TFixedImage, TMovingImage >
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
    PAverageNormalizedMutualInformationMetric< TFixedImage, TMovingImage >
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
    PAverageNormalizedMutualInformationMetric< TFixedImage, TMovingImage >
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
    PAverageNormalizedMutualInformationMetric< TFixedImage, TMovingImage >
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
    PAverageNormalizedMutualInformationMetric<TFixedImage,TMovingImage>
    ::ComputeValueAndPRatioArray( double & nMI, unsigned int n ) const
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
        
        double sumnum = 0.0;
        double sumden = 0.0;
        while( fixedPDFit != fixedPDFend )
        {
            const double logFixedImagePDFValue = *fixedPDFit;
            
            movingPDFit = movingPDFbegin;
            
            while( movingPDFit != movingPDFend )
            {
                const PDFValueType logMovingImagePDFValue = *movingPDFit;
                const PDFValueType jointPDFValue  = jointPDFit.Value();
                
                sumnum -= jointPDFValue * ( logFixedImagePDFValue + logMovingImagePDFValue );

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
                const PDFValueType logMovingImagePDFValue = *movingPDFit;
                const PDFValueType jointPDFValue  = jointPDFit.Value();
                
                if( jointPDFValue > 1e-16 )
                {
                    const PDFValueType pRatio = (nMI * vcl_log(jointPDFValue) - logFixedImagePDFValue -logMovingImagePDFValue) / sumden;
                    (*(this->m_PRatioArray[this->m_RandomList[n]]))[ fixedIndex ][ movingIndex ] = static_cast< PRatioType >(pRatio / this->m_NumberOfPixelsCountedVector[this->m_RandomList[n]] / static_cast<float>(this->m_NumSamplesLastDimension));
                    
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
}

#endif // end #ifndef _itkPAverageNormalizedMutualInformationMetric_HXX__
