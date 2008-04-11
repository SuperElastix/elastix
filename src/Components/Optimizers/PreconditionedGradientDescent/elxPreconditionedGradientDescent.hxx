#ifndef __elxPreconditionedGradientDescent_hxx
#define __elxPreconditionedGradientDescent_hxx

#include "elxPreconditionedGradientDescent.h"
#include <iomanip>
#include <string>

/** \todo: temp hack */
#include "../../Metrics/AdvancedMeanSquares/itkAdvancedMeanSquaresImageToImageMetric.h"


namespace elastix
{
using namespace itk;


  /**
	 * ***************** Constructor ***********************
	 */

	template <class TElastix>
		PreconditionedGradientDescent<TElastix>::
		PreconditionedGradientDescent()
	{
    this->m_MaximumNumberOfSamplingAttempts = 0;
    this->m_CurrentNumberOfSamplingAttempts = 0;
    this->m_PreviousErrorAtIteration = 0;
    this->m_PreconditionMatrixSet = false;    
    this->m_SelfHessianSmoothingSigma = 1.0;
    this->m_NumberOfSamplesForSelfHessian = 100000;

  } // end Constructor()


	/**
	 * ***************** BeforeRegistration ***********************
	 */

	template <class TElastix>
		void PreconditionedGradientDescent<TElastix>::
		BeforeRegistration( void )
	{
		/** Add the target cell "stepsize" to xout["iteration"].*/
		xout["iteration"].AddTargetCell("2:Metric");
		xout["iteration"].AddTargetCell("3:StepSize");
		xout["iteration"].AddTargetCell("4:||Gradient||");

		/** Format the metric and stepsize as floats */			
		xl::xout["iteration"]["2:Metric"]		<< std::showpoint << std::fixed;
		xl::xout["iteration"]["3:StepSize"] << std::showpoint << std::fixed;
		xl::xout["iteration"]["4:||Gradient||"] << std::showpoint << std::fixed;

	} // end BeforeRegistration()


	/**
	 * ***************** BeforeEachResolution ***********************
	 */

	template <class TElastix>
		void PreconditionedGradientDescent<TElastix>
		::BeforeEachResolution(void)
	{
		/** Get the current resolution level. */
		unsigned int level = static_cast<unsigned int>(
			this->m_Registration->GetAsITKBaseType()->GetCurrentLevel() );
				
		/** Set the maximumNumberOfIterations. */
		unsigned int maximumNumberOfIterations = 500;
		this->GetConfiguration()->ReadParameter( maximumNumberOfIterations,
      "MaximumNumberOfIterations", this->GetComponentLabel(), level, 0 );
		this->SetNumberOfIterations( maximumNumberOfIterations );

    /** Set the gain parameters */
		double a = 400.0;
		double A = 50.0;
		double alpha = 0.602;

		this->GetConfiguration()->ReadParameter(a, "SP_a", this->GetComponentLabel(), level, 0 );
		this->GetConfiguration()->ReadParameter(A, "SP_A", this->GetComponentLabel(), level, 0 );
		this->GetConfiguration()->ReadParameter(alpha, "SP_alpha", this->GetComponentLabel(), level, 0 );
		
		this->SetParam_a(	a );
		this->SetParam_A( A );
		this->SetParam_alpha( alpha );
  
    /** Set the MaximumNumberOfSamplingAttempts. */
		unsigned int maximumNumberOfSamplingAttempts = 0;
		this->GetConfiguration()->ReadParameter( maximumNumberOfSamplingAttempts,
      "MaximumNumberOfSamplingAttempts", this->GetComponentLabel(), level, 0 );
		this->SetMaximumNumberOfSamplingAttempts( maximumNumberOfSamplingAttempts );

    /** Set the minimum condition number for the precondition matrix */
    double minimumConditionNumber = 0.01;
    this->GetConfiguration()->ReadParameter( minimumConditionNumber,
      "MinimumConditionNumber", this->GetComponentLabel(), level, 0 );
    this->SetMinimumConditionNumber( minimumConditionNumber );
				
	} // end BeforeEachResolution()


	/**
	 * ***************** AfterEachIteration *************************
	 */

	template <class TElastix>
		void PreconditionedGradientDescent<TElastix>
		::AfterEachIteration(void)
	{
		/** Print some information */
		xl::xout["iteration"]["2:Metric"]		<< this->GetValue();
		xl::xout["iteration"]["3:StepSize"] << this->GetLearningRate();
		xl::xout["iteration"]["4:||Gradient||"] << this->GetGradient().magnitude();

		/** Select new spatial samples for the computation of the metric */
		if ( this->GetNewSamplesEveryIteration() )
		{
			this->SelectNewSamples();
		}

	} // end AfterEachIteration()


	/**
	 * ***************** AfterEachResolution *************************
	 */

	template <class TElastix>
		void PreconditionedGradientDescent<TElastix>
		::AfterEachResolution( void )
	{
		/**
		 * enum   StopConditionType {  MaximumNumberOfIterations, MetricError }  
		 */
		std::string stopcondition;
		switch ( this->GetStopCondition() )
		{
	
		case MaximumNumberOfIterations :
			stopcondition = "Maximum number of iterations has been reached";	
			break;	
		
		case MetricError :
			stopcondition = "Error in metric";	
			break;	
				
		default:
			stopcondition = "Unknown";
			break;
			
		}

		/** Print the stopping condition */
		elxout << "Stopping condition: " << stopcondition << "." << std::endl;

	} // end AfterEachResolution()
	

	/**
	 * ******************* AfterRegistration ************************
	 */

	template <class TElastix>
		void PreconditionedGradientDescent<TElastix>
		::AfterRegistration(void)
	{
	  /** Print the best metric value */
		double bestValue = this->GetValue();
		elxout
			<< std::endl
			<< "Final metric value  = " 
			<< bestValue
			<< std::endl;
		
	} // end AfterRegistration()


  /**
   * ****************** StartOptimization *************************
   */

  template <class TElastix>
    void PreconditionedGradientDescent<TElastix>
    ::StartOptimization( void )
	{
		/** Check if the entered scales are correct and != [ 1 1 1 ...] */
		this->SetUseScales( false );
		const ScalesType & scales = this->GetScales();
		if ( scales.GetSize() == this->GetInitialPosition().GetSize() )
		{
      ScalesType unit_scales( scales.GetSize() );
			unit_scales.Fill(1.0);
			if ( scales != unit_scales )
			{
				/** only then: */
				this->SetUseScales( true );
			}
		}

    /** Reset these values. */
    this->m_CurrentNumberOfSamplingAttempts = 0;
    this->m_PreviousErrorAtIteration = 0;
    this->m_PreconditionMatrixSet = false;

    /** Superclass implementation. */
		this->Superclass1::StartOptimization();

	} // end StartOptimization()

 /** 
  * ********************** ResumeOptimization **********************
  */

  template <class TElastix>
    void PreconditionedGradientDescent<TElastix>
    ::ResumeOptimization( void )
  {
    if ( !this->m_PreconditionMatrixSet )
    {
      this->SetSelfHessian();
      // hack
      this->m_PreconditionMatrixSet = true;      
    }

    this->Superclass1::ResumeOptimization();

  } // end ResumeOptimization()


  /** 
  * ********************** SetSelfHessian **********************
  */

  template <class TElastix>
    void PreconditionedGradientDescent<TElastix>
    ::SetSelfHessian( void )
  {
    /** If it works, think about a more generic solution */
    typedef typename RegistrationType::FixedImageType   FixedImageType;
    typedef typename RegistrationType::MovingImageType  MovingImageType;
    typedef AdvancedMeanSquaresImageToImageMetric<
      FixedImageType, MovingImageType>                  MetricWithSelfHessianType;
    typedef typename MetricWithSelfHessianType::Pointer MetricWithSelfHessianPointer;

    PreconditionType H;

    /* Get metric as metric with self hessian. 
     * \todo Does not work for multimetric yet! */
    MetricWithSelfHessianPointer metricWithSelfHessian = dynamic_cast<
      MetricWithSelfHessianType *>( this->GetElastix()->GetElxMetricBase() );

    if ( metricWithSelfHessian.IsNull() )
    {
      itkExceptionMacro( << "The PreconditionedGradientDescent optimizer can only be used with the AdvancedMeanSquares metric!" );
    }
    
    elxout << "Computing SelfHessian." << std::endl;
    try
    {
      metricWithSelfHessian->GetSelfHessian( this->GetCurrentPosition(), H );
    }
    catch(ExceptionObject & err)
    {
      this->m_StopCondition = MetricError;
      this->StopOptimization();
      throw err;
    }

    elxout << "Computing eigensystem of SelfHessian." << std::endl;
    this->SetPreconditionMatrix( H );

    unsigned int nrOfEigenModesRetained = this->GetEigenSystem()->D.cols();
    elxout << "Number of eigen modes retained / number of parameters: " << 
     nrOfEigenModesRetained  << "/" << H.cols() << std::endl;
    elxout << "First and last inverted eigenvalues: " 
      << this->GetEigenSystem()->D[ 0 ]
      << ", "
      << this->GetEigenSystem()->D[ nrOfEigenModesRetained - 1 ]
      << std::endl;

  } // end SetSelfHessian()

		

  /**
   * ****************** MetricErrorResponse *************************
   */

  template <class TElastix>
    void PreconditionedGradientDescent<TElastix>
    ::MetricErrorResponse( ExceptionObject & err )
	{
    if ( this->GetCurrentIteration() != this->m_PreviousErrorAtIteration )
    {
      this->m_PreviousErrorAtIteration = this->GetCurrentIteration();
      this->m_CurrentNumberOfSamplingAttempts = 1;
    }
    else
    {
      this->m_CurrentNumberOfSamplingAttempts++;
    }

    if ( this->m_CurrentNumberOfSamplingAttempts <= this->m_MaximumNumberOfSamplingAttempts )
    {
      this->SelectNewSamples();
      this->ResumeOptimization();
    }
    else
    {
      /** Stop optimisation and pass on exception. */
      this->Superclass1::MetricErrorResponse( err );
    }

  } // end MetricErrorResponse()


} // end namespace elastix

#endif // end #ifndef __elxPreconditionedGradientDescent_hxx

