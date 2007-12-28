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

    /** Normalize the pdfs: p = alpha h  */
    this->NormalizeJointPDF( this->m_JointPDF, this->m_Alpha );
    
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
    double MI = 0.0;
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
          MI += jointPDFValue * vcl_log( jointPDFValue / fixPDFmovPDF );
        }  
        ++movingPDFit;
        ++jointPDFit;
      }  // end while-loop over moving index
      ++fixedPDFit;
      jointPDFit.NextLine();
    }  // end while-loop over fixed index
    
    return static_cast<MeasureType>( -1.0 * MI );
    
  } // end GetValue


	/**
	 * ******************** GetValueAndAnalyticDerivative *******************
	 * Get both the Value and the Derivative of the Measure. 
	 */

	template < class TFixedImage, class TMovingImage  >
	  void
	  ParzenWindowMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
	  ::GetValueAndAnalyticDerivative(
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

    /** Normalize the pdfs: p = alpha h  */
    this->NormalizeJointPDF( this->m_JointPDF, this->m_Alpha  );
    
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
    const DerivativeIteratorType derivbegin = derivative.begin();
    const DerivativeIteratorType derivend = derivative.end();
    
    double MI = 0.0;
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
          derivit = derivbegin;
          const double pRatio = vcl_log( jointPDFValue / fixPDFmovPDF );
          const double pRatioAlpha = this->m_Alpha * pRatio;
          MI += jointPDFValue * pRatio;
          while ( derivit != derivend )
          {                    
            /**  Ref: eqn 23 of Thevenaz & Unser paper [3] */
            (*derivit) -= jointPDFDerivativesit.Get() * pRatioAlpha;
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
    
    value = static_cast<MeasureType>( -1.0 * MI);
       
  } // end GetValueAndAnalyticDerivative


  /**
	 * ******************** GetValueAndFiniteDifferenceDerivative *******************
	 * Get both the Value and the Derivative of the Measure. 
   * Compute the derivative using a finite difference approximation
	 */

	template < class TFixedImage, class TMovingImage  >
	  void
	  ParzenWindowMutualInformationImageToImageMetric<TFixedImage,TMovingImage>
	  ::GetValueAndFiniteDifferenceDerivative(
	  const ParametersType& parameters,
	  MeasureType& value,
	  DerivativeType& derivative) const
	{		 
    /** Initialize some variables */
    value = NumericTraits< MeasureType >::Zero;
    derivative = DerivativeType( this->GetNumberOfParameters() );
    derivative.Fill( NumericTraits<double>::Zero );

    /** Construct the JointPDF, JointPDFDerivatives, Alpha and its derivatives. */
    this->ComputePDFsAndIncrementalPDFs( parameters );
    
    /** Compute the fixed and moving marginal pdf by summing over the histogram */
    this->ComputeMarginalPDF( this->m_JointPDF, this->m_FixedImageMarginalPDF, 0 );
    this->ComputeMarginalPDF( this->m_JointPDF, this->m_MovingImageMarginalPDF, 1 );

    /** Compute the fixed and moving incremental marginal pdfs by summing over the 
     * incremental histogram. Do it for Right and Left. */
    this->ComputeIncrementalMarginalPDFs( this->m_IncrementalJointPDFRight,
      this->m_FixedIncrementalMarginalPDFRight, this->m_MovingIncrementalMarginalPDFRight );
    this->ComputeIncrementalMarginalPDFs( this->m_IncrementalJointPDFLeft,
      this->m_FixedIncrementalMarginalPDFLeft, this->m_MovingIncrementalMarginalPDFLeft );

    /** Compute the metric and derivatives by double summation over histogram. */

    /** Setup iterators */
    typedef ImageLinearConstIteratorWithIndex<
      JointPDFType >                                 JointPDFIteratorType;
    typedef ImageLinearConstIteratorWithIndex<
      JointPDFDerivativesType>                       IncrementalJointPDFIteratorType;
    typedef typename MarginalPDFType::const_iterator MarginalPDFIteratorType;
    typedef ImageLinearConstIteratorWithIndex<
      IncrementalMarginalPDFType >                   IncrementalMarginalPDFIteratorType;
    typedef typename DerivativeType::iterator        DerivativeIteratorType;
    typedef typename DerivativeType::const_iterator  DerivativeConstIteratorType;
 
    JointPDFIteratorType jointPDFit(
      this->m_JointPDF, this->m_JointPDF->GetLargestPossibleRegion() );
    jointPDFit.GoToBegin();    

    IncrementalJointPDFIteratorType jointIncPDFRightit( this->m_IncrementalJointPDFRight, 
      this->m_IncrementalJointPDFRight->GetLargestPossibleRegion() );
    IncrementalJointPDFIteratorType jointIncPDFLeftit( this->m_IncrementalJointPDFLeft, 
      this->m_IncrementalJointPDFLeft->GetLargestPossibleRegion() );
    jointIncPDFRightit.GoToBegin();
    jointIncPDFLeftit.GoToBegin();
    
    MarginalPDFIteratorType fixedPDFit = this->m_FixedImageMarginalPDF.begin();
    const MarginalPDFIteratorType fixedPDFend = this->m_FixedImageMarginalPDF.end();
    MarginalPDFIteratorType movingPDFit = this->m_MovingImageMarginalPDF.begin();
    const MarginalPDFIteratorType movingPDFend = this->m_MovingImageMarginalPDF.end();

    IncrementalMarginalPDFIteratorType fixedIncPDFRightit(
      this->m_FixedIncrementalMarginalPDFRight,
      this->m_FixedIncrementalMarginalPDFRight->GetLargestPossibleRegion() );
    IncrementalMarginalPDFIteratorType movingIncPDFRightit(
      this->m_MovingIncrementalMarginalPDFRight,
      this->m_MovingIncrementalMarginalPDFRight->GetLargestPossibleRegion() );
    IncrementalMarginalPDFIteratorType fixedIncPDFLeftit(
      this->m_FixedIncrementalMarginalPDFLeft,
      this->m_FixedIncrementalMarginalPDFLeft->GetLargestPossibleRegion() );
    IncrementalMarginalPDFIteratorType movingIncPDFLeftit(
      this->m_MovingIncrementalMarginalPDFLeft,
      this->m_MovingIncrementalMarginalPDFLeft->GetLargestPossibleRegion() );
    fixedIncPDFRightit.GoToBegin();
    movingIncPDFRightit.GoToBegin();
    fixedIncPDFLeftit.GoToBegin();
    movingIncPDFLeftit.GoToBegin();
    
    DerivativeIteratorType derivit = derivative.begin();
    const DerivativeIteratorType derivbegin = derivative.begin();
    const DerivativeIteratorType derivend = derivative.end();

    DerivativeConstIteratorType perturbedAlphaRightit = this->m_PerturbedAlphaRight.begin();
    const DerivativeConstIteratorType perturbedAlphaRightbegin = this->m_PerturbedAlphaRight.begin();
    DerivativeConstIteratorType perturbedAlphaLeftit = this->m_PerturbedAlphaLeft.begin();
    const DerivativeConstIteratorType perturbedAlphaLeftbegin = this->m_PerturbedAlphaLeft.begin();
    
    double MI = 0.0;
    while ( fixedPDFit != fixedPDFend )
    {
      const double fixedPDFValue = *fixedPDFit;
      
      while ( movingPDFit != movingPDFend )
      {
        const double movingPDFValue = *movingPDFit;
        const double jointPDFValue = jointPDFit.Get();
        const double fixPDFmovPDFAlpha = 
          fixedPDFValue * movingPDFValue * this->m_Alpha;

        /** check for non-zero bin contribution and update the mutual information value */
        if( jointPDFValue > 1e-16 && fixPDFmovPDFAlpha > 1e-16 )
        {
          MI += this->m_Alpha * jointPDFValue * vcl_log( jointPDFValue / fixPDFmovPDFAlpha );
        }

        /** Update the derivative */
        derivit = derivbegin;
        perturbedAlphaRightit = perturbedAlphaRightbegin;
        perturbedAlphaLeftit = perturbedAlphaLeftbegin;
        while ( derivit != derivend )
        { 
          /** Initialize */
          double contrib = 0.0;

          /** For clarity, get some values and give them a name. 
           * \todo Does this cost a lot of computation time? */
          const double jointIncPDFRightValue = jointIncPDFRightit.Get();
          const double fixedIncPDFRightValue = fixedIncPDFRightit.Get();
          const double movingIncPDFRightValue = movingIncPDFRightit.Get();
          const double perturbedAlphaRightValue = *perturbedAlphaRightit;
          
          /** Compute the contribution of the Right-perturbation to the derivative */
          const double perturbedJointPDFRightValue = jointIncPDFRightValue + jointPDFValue;
          const double perturbedFixedPDFRightValue = fixedPDFValue + fixedIncPDFRightValue;
          const double perturbedMovingPDFRightValue = movingPDFValue + movingIncPDFRightValue;
          const double perturbedfixPDFmovPDFAlphaRight =
            perturbedFixedPDFRightValue * perturbedMovingPDFRightValue * perturbedAlphaRightValue;
          if ( perturbedJointPDFRightValue > 1e-16 && perturbedfixPDFmovPDFAlphaRight > 1e-16 )
          { 
            contrib += perturbedAlphaRightValue * perturbedJointPDFRightValue *
              vcl_log( perturbedJointPDFRightValue / perturbedfixPDFmovPDFAlphaRight );
          }

          /** For clarity, get some values and give them a name */
          const double jointIncPDFLeftValue = jointIncPDFLeftit.Get();
          const double fixedIncPDFLeftValue = fixedIncPDFLeftit.Get();
          const double movingIncPDFLeftValue = movingIncPDFLeftit.Get();
          const double perturbedAlphaLeftValue = *perturbedAlphaLeftit;
          
          /** Compute the contribution of the Left-perturbation to the derivative */
          const double perturbedJointPDFLeftValue = jointIncPDFLeftValue + jointPDFValue;
          const double perturbedFixedPDFLeftValue = fixedPDFValue + fixedIncPDFLeftValue;
          const double perturbedMovingPDFLeftValue = movingPDFValue + movingIncPDFLeftValue;
          const double perturbedfixPDFmovPDFAlphaLeft =
            perturbedFixedPDFLeftValue * perturbedMovingPDFLeftValue * perturbedAlphaLeftValue;
          if ( perturbedJointPDFLeftValue > 1e-16 && perturbedfixPDFmovPDFAlphaLeft > 1e-16 )
          { 
            contrib -= perturbedAlphaLeftValue * perturbedJointPDFLeftValue *
              vcl_log( perturbedJointPDFLeftValue / perturbedfixPDFmovPDFAlphaLeft );
          }          

          /** Update the derivative component */
          (*derivit) += contrib;

          /** move the iterators to the next parameter */
          ++derivit;
          ++perturbedAlphaRightit; 
          ++perturbedAlphaLeftit;
          ++jointIncPDFRightit;
          ++jointIncPDFLeftit;
          ++fixedIncPDFRightit;
          ++movingIncPDFRightit;
          ++fixedIncPDFLeftit;
          ++movingIncPDFLeftit;
        }  // end while-loop over parameters
        
        ++jointPDFit; // next moving bin
        ++movingPDFit; // next moving bin        
        jointIncPDFRightit.NextLine(); // next moving bin
        jointIncPDFLeftit.NextLine(); // next moving bin
        fixedIncPDFRightit.GoToBeginOfLine(); // same fixed bin
        fixedIncPDFLeftit.GoToBeginOfLine(); // same fixed bin
        movingIncPDFRightit.NextLine(); // next moving bin
        movingIncPDFLeftit.NextLine(); // next moving bin

      }  // end while-loop over moving index

      jointPDFit.NextLine(); // next fixed bin
      ++fixedPDFit; // next fixed bin
      movingPDFit = this->m_MovingImageMarginalPDF.begin(); // first moving bin
      fixedIncPDFRightit.NextLine(); // next fixed bin
      fixedIncPDFLeftit.NextLine(); // next fixed bin
      movingIncPDFRightit.GoToBegin(); // first moving bin
      movingIncPDFLeftit.GoToBegin(); // first moving bin

    }  // end while-loop over fixed index
    
    value = static_cast<MeasureType>( -1.0 * MI);

    /** Divide the derivative by -delta*2 */            
    const double delta2 = - 1.0 / ( this->GetFiniteDifferencePerturbation() * 2.0 );
    derivit = derivative.begin();
    while ( derivit != derivend )
    {                    
      (*derivit) *= delta2;
      ++derivit;
    }
        
  } // end GetValueAndFiniteDifferenceDerivative

} // end namespace itk 


#endif // end #ifndef _itkParzenWindowMutualInformationImageToImageMetric_HXX__

