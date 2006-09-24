#ifndef _itkParzenWindowMutualInformationImageToImageMetric_HXX__
#define _itkParzenWindowMutualInformationImageToImageMetric_HXX__

#include "itkParzenWindowMutualInformationImageToImageMetric.h"

#include "itkImageLinearConstIteratorWithIndex.h"
#include "vnl/vnl_math.h"

namespace itk
{	
	
	/**
	 * ********************* PrintSelf ******************************
	 *
	 * Print out internal information about this class.
	 */

	template < class TFixedImage, class TMovingImage  >
		void
		ParzenWindowMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
		::PrintSelf( std::ostream& os, Indent indent ) const
	{
		/** Call the superclass' PrintSelf. */
		Superclass::PrintSelf( os, indent );
	
    /** This function is not complete, but we don't use it anyway. */
		
	} // end PrintSelf
	
	
  /**
	 * ************************** GetValue **************************
	 * Get the match Measure.
	 */

	template < class TFixedImage, class TMovingImage  >
	  typename ParzenWindowMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
	  ::MeasureType
	  ParzenWindowMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
	  ::GetValue( const ParametersType& parameters ) const
	{		 
    /** Construct the JointPDF and Alpha */
    this->ComputePDFs(parameters);          

    /** Normalize the pdfs: p = alpha h / e_T e_R */
    this->NormalizeJointPDF( this->m_JointPDF, 
      this->m_Alpha / this->m_FixedImageBinSize / this->m_MovingImageBinSize );
    
    /** Compute the fixed and moving marginal pdfs, by summing over the joint pdf */
    this->ComputeMarginalPDF( this->m_JointPDF, this->m_FixedImageMarginalPDF, 0 );
    this->ComputeMarginalPDF( this->m_JointPDF, this->m_MovingImageMarginalPDF, 1 );

    /** Compute the metric by double summation over histogram. */

    /** Setup iterators */
    typedef ImageLinearConstIteratorWithIndex<JointPDFType> JointPDFIteratorType;
    typedef typename MarginalPDFType::const_iterator        MarginalPDFIteratorType;

    JointPDFIteratorType jointPDFit(
      this->m_JointPDF, this->m_JointPDF->GetLargestPossibleRegion() );
    jointPDFit.SetDirection(0);
    jointPDFit.GoToBegin();    
    MarginalPDFIteratorType fixedPDFit = this->m_FixedImageMarginalPDF.begin();
    const MarginalPDFIteratorType fixedPDFend = this->m_FixedImageMarginalPDF.end();
    MarginalPDFIteratorType movingPDFit = this->m_MovingImageMarginalPDF.begin();
    const MarginalPDFIteratorType movingPDFend = this->m_MovingImageMarginalPDF.end();
       
    /** Loop over histogram */
    double sum = 0.0;
    while ( fixedPDFit != fixedPDFend )
    {
      const double fixedImagePDFValue = *fixedPDFit;
      movingPDFit = this->m_MovingImageMarginalPDF.begin();
      while ( movingPDFit != movingPDFend )
      {
        const double movingImagePDFValue = *movingPDFit;
        const double fixPDFmovPDF = fixedImagePDFValue * movingImagePDFValue;
        const double jointPDFValue = jointPDFit.Get();
        /** check for non-zero bin contribution */
        if( jointPDFValue > 1e-16 && fixPDFmovPDF > 1e-16 )
        {
          sum += jointPDFValue * vcl_log( jointPDFValue / fixPDFmovPDF );
        }  
        ++movingPDFit;
        ++jointPDFit;
      }  // end while-loop over moving index
      ++fixedPDFit;
      jointPDFit.NextLine();
    }  // end while-loop over fixed index
    
    return static_cast<MeasureType>( -1.0 * sum );
    
  } // end GetValue


	/**
	 * ******************** GetValueAndDerivative *******************
	 * Get both the Value and the Derivative of the Measure. 
	 */

	template < class TFixedImage, class TMovingImage  >
	  void
	  ParzenWindowMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
	  ::GetValueAndDerivative(
	  const ParametersType& parameters,
	  MeasureType& value,
	  DerivativeType& derivative) const
	{		 
    /** Initialize some variables */
    value = NumericTraits< MeasureType >::Zero;
    derivative = DerivativeType( this->GetNumberOfParameters() );
    derivative.Fill( NumericTraits<double>::Zero );

    /** Construct the JointPDF, JointPDFDerivatives, Alpha and its derivatives. */
    this->ComputePDFsAndPDFDerivatives( parameters );

    /** Normalize the pdfs: p = alpha h / e_T e_R */
    this->NormalizeJointPDF(
      this->m_JointPDF,
      this->m_Alpha / this->m_FixedImageBinSize / this->m_MovingImageBinSize );
    this->NormalizeJointPDFDerivatives(
      this->m_JointPDFDerivatives, 
      this->m_Alpha / this->m_FixedImageBinSize / this->m_MovingImageBinSize );

    /** Compute the fixed and moving marginal pdf by summing over the histogram */
    this->ComputeMarginalPDF( this->m_JointPDF, this->m_FixedImageMarginalPDF, 0 );
    this->ComputeMarginalPDF( this->m_JointPDF, this->m_MovingImageMarginalPDF, 1 );

    /** Compute the metric and derivatives by double summation over histogram. */

    /** Setup iterators */
    typedef ImageLinearConstIteratorWithIndex<
      JointPDFType >                                 JointPDFIteratorType;
    typedef ImageLinearConstIteratorWithIndex<
      JointPDFDerivativesType>                       JointPDFDerivativesIteratorType;
    typedef typename MarginalPDFType::const_iterator MarginalPDFIteratorType;
    typedef typename DerivativeType::iterator        DerivativeIteratorType;
    typedef typename DerivativeType::const_iterator  DerivativeConstIteratorType;
 
    JointPDFIteratorType jointPDFit(
      this->m_JointPDF, this->m_JointPDF->GetLargestPossibleRegion() );
    jointPDFit.SetDirection(0);
    jointPDFit.GoToBegin();    
    JointPDFDerivativesIteratorType jointPDFDerivativesit(
      this->m_JointPDFDerivatives, this->m_JointPDFDerivatives->GetLargestPossibleRegion() );
    jointPDFDerivativesit.SetDirection(0);
    jointPDFDerivativesit.GoToBegin();    
    MarginalPDFIteratorType fixedPDFit = this->m_FixedImageMarginalPDF.begin();
    const MarginalPDFIteratorType fixedPDFend = this->m_FixedImageMarginalPDF.end();
    MarginalPDFIteratorType movingPDFit = this->m_MovingImageMarginalPDF.begin();
    const MarginalPDFIteratorType movingPDFend = this->m_MovingImageMarginalPDF.end();
    DerivativeIteratorType derivit = derivative.begin();
    const DerivativeConstIteratorType derivend = derivative.end();
    
    double sum = 0.0;
    while ( fixedPDFit != fixedPDFend )
    {
      const double fixedImagePDFValue = *fixedPDFit;
      movingPDFit = this->m_MovingImageMarginalPDF.begin();
      while ( movingPDFit != movingPDFend )
      {
        const double movingImagePDFValue = *movingPDFit;
        const double fixPDFmovPDF = fixedImagePDFValue * movingImagePDFValue;
        const double jointPDFValue = jointPDFit.Get();
        /** check for non-zero bin contribution */
        if( jointPDFValue > 1e-16 && fixPDFmovPDF > 1e-16 )
        {
          derivit = derivative.begin();
          const double pRatio = vcl_log( jointPDFValue / fixPDFmovPDF );
          sum += jointPDFValue * pRatio;
          while ( derivit != derivend )
          {                    
            /**  Ref: eqn 23 of Thevenaz & Unser paper [3] */
            (*derivit) -= jointPDFDerivativesit.Get() * pRatio;
            ++derivit;
            ++jointPDFDerivativesit;
          }  // end while-loop over parameters
        }  // end if-block to check non-zero bin contribution
        ++movingPDFit;
        ++jointPDFit;
        jointPDFDerivativesit.NextLine();
      }  // end while-loop over moving index
      ++fixedPDFit;
      jointPDFit.NextLine();
    }  // end while-loop over fixed index
    
    value = static_cast<MeasureType>( -1.0 * sum );

    /** Add -1/alpha * dalpha/dmu * sum_i sum_k p log( p / pT pR) ) 
     * sum = sum_i sum_k alpha p log( p / (p / pT pR) )
     * so we have divide by -alpha */            
    const double alphaDerivativeFactor = - sum / this->m_Alpha;
    derivit = derivative.begin();
    DerivativeConstIteratorType alphaDerivit = this->m_AlphaDerivatives.begin();
    while ( derivit != derivend )
    {                    
      (*derivit) += (*alphaDerivit) * alphaDerivativeFactor;
      ++derivit;
      ++alphaDerivit;
    }
        
  } // end GetValueAndDerivative


} // end namespace itk 


#endif // end #ifndef _itkParzenWindowMutualInformationImageToImageMetric_HXX__

