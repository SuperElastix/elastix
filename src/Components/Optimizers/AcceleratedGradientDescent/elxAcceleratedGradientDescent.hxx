#ifndef __elxAcceleratedGradientDescent_hxx
#define __elxAcceleratedGradientDescent_hxx

#include "elxAcceleratedGradientDescent.h"
#include <iomanip>
#include <string>
#include <vector>
#include "vnl/vnl_math.h"
#include "itkImageRandomConstIteratorWithIndex.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"

namespace elastix
{
using namespace itk;

	/**
	 * ********************** Constructor ***********************
	 */

	template <class TElastix>
		AcceleratedGradientDescent<TElastix>::
		AcceleratedGradientDescent()
  {
    this->m_AutomaticGainEstimation = false;
    this->m_InitialStepSize = 1.0;
  } // Constructor


	/**
	 * ***************** BeforeRegistration ***********************
	 */

	template <class TElastix>
		void AcceleratedGradientDescent<TElastix>::
		BeforeRegistration(void)
	{
		
		/** Add the target cell "stepsize" to xout["iteration"].*/
		xout["iteration"].AddTargetCell("2:Metric");
		xout["iteration"].AddTargetCell("3:StepSize");
		xout["iteration"].AddTargetCell("4:||Gradient||");

		/** Format the metric and stepsize as floats */			
		xl::xout["iteration"]["2:Metric"]		<< std::showpoint << std::fixed;
		xl::xout["iteration"]["3:StepSize"] << std::showpoint << std::fixed;
		xl::xout["iteration"]["4:||Gradient||"] << std::showpoint << std::fixed;

	} // end BeforeRegistration


	/**
	 * ***************** BeforeEachResolution ***********************
	 */

	template <class TElastix>
		void AcceleratedGradientDescent<TElastix>
		::BeforeEachResolution(void)
	{
		/** Get the current resolution level.*/
		unsigned int level = static_cast<unsigned int>(
			this->m_Registration->GetAsITKBaseType()->GetCurrentLevel() );
				
		/** Set the maximumNumberOfIterations.*/
		unsigned int maximumNumberOfIterations = 100;
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

    /** Set/Get whether CruzzAcceleration is desired. Default: false */
    bool usecruz = false;
		this->GetConfiguration()->ReadParameter( usecruz,
      "UseCruzAcceleration", this->GetComponentLabel(), level, 0 );
		this->SetUseCruzAcceleration( usecruz );
    
    if ( usecruz )
    {
      /** Set/Get the maximum of the sigmoid use by CruzAcceleration. 
      * Should be >0. Default: 1.0 */     
      double sigmoidMax = 1.0;
      this->GetConfiguration()->ReadParameter( sigmoidMax,
        "SigmoidMax", this->GetComponentLabel(), level, 0 );
      this->SetSigmoidMax( sigmoidMax );

      /** Set/Get the maximum of the sigmoid use by CruzAcceleration. 
      * Should be <0. Default: -0.999 */     
      double sigmoidMin = -0.999;
      this->GetConfiguration()->ReadParameter( sigmoidMin,
        "SigmoidMin", this->GetComponentLabel(), level, 0 );
      this->SetSigmoidMin( sigmoidMin );

      /** Set/Get the scaling of the sigmoid width. Large values 
      * cause a more wide sigmoid. Default: 1e-8. Should be >0. */     
      double sigmoidScale = 1e-8;
      this->GetConfiguration()->ReadParameter( sigmoidScale,
        "SigmoidScale", this->GetComponentLabel(), level, 0 );
      this->SetSigmoidScale( sigmoidScale );

      /** Set/Get the initial time. Default: 10.0. Should be >0. */     
      double initialTime = 10.0;
      this->GetConfiguration()->ReadParameter( initialTime,
        "SigmoidInitialTime", this->GetComponentLabel(), level, 0 );
      this->SetInitialTime( initialTime );
    }

    /** Set whether automatic gain estimation is required */
    this->m_AutomaticGainEstimation = false;
    this->GetConfiguration()->ReadParameter( this->m_AutomaticGainEstimation,
      "AutomaticGainEstimation", this->GetComponentLabel(), level, 0 );

    if ( this->m_AutomaticGainEstimation )
    {
      /** Set the initial step size: the desired displacement of a voxel in mm,
       * at the first iteration. */
      this->m_InitialStepSize = 1.0;
      this->GetConfiguration()->ReadParameter( this->m_InitialStepSize,
        "InitialStepSize", this->GetComponentLabel(), level, 0 );
    }

				
	} // end BeforeEachResolution


	/**
	 * ***************** AfterEachIteration *************************
	 */

	template <class TElastix>
		void AcceleratedGradientDescent<TElastix>
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

	} // end AfterEachIteration


	/**
	 * ***************** AfterEachResolution *************************
	 */

	template <class TElastix>
		void AcceleratedGradientDescent<TElastix>
		::AfterEachResolution(void)
	{
		
		/**
		 * enum   StopConditionType {  MaximumNumberOfIterations, MetricError }  
		 */
		std::string stopcondition;

		
		switch( this->GetStopCondition() )
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

	} // end AfterEachResolution
	

	/**
	 * ******************* AfterRegistration ************************
	 */

	template <class TElastix>
		void AcceleratedGradientDescent<TElastix>
		::AfterRegistration(void)
	{
	  /** Print the best metric value */
		
		double bestValue = this->GetValue();
		elxout
			<< std::endl
			<< "Final metric value  = " 
			<< bestValue
			<< std::endl;
		
	} // end AfterRegistration


  /**
   * ****************** StartOptimization *************************
   */

  template <class TElastix>
    void AcceleratedGradientDescent<TElastix>
    ::StartOptimization(void)
	{

		/** Check if the entered scales are correct and != [ 1 1 1 ...] */

		this->SetUseScales(false);
		const ScalesType & scales = this->GetScales();
		if ( scales.GetSize() == this->GetInitialPosition().GetSize() )
		{
      ScalesType unit_scales( scales.GetSize() );
			unit_scales.Fill(1.0);
			if (scales != unit_scales)
			{
				/** only then: */
				this->SetUseScales(true);
			}
		}

		this->Superclass1::StartOptimization();

	} //end StartOptimization
	

  /** 
   * ********************** AdvanceOneStep **********************
   */

    template <class TElastix>
    void AcceleratedGradientDescent<TElastix>
    ::AdvanceOneStep(void)
	{
    typedef typename RegistrationType::FixedImageType   FixedImageType;
    typedef typename FixedImageType::RegionType         FixedImageRegionType;
    typedef typename FixedImageType::IndexType          FixedImageIndexType;
    typedef typename FixedImageType::PointType          FixedImagePointType;
    typedef typename RegistrationType::ITKBaseType      itkRegistrationType;
    typedef typename itkRegistrationType::TransformType TransformType;
    typedef typename TransformType::JacobianType        JacobianType;
    typedef itk::ImageRandomConstIteratorWithIndex<FixedImageType>
                                                        FixedImageIteratorType;
    typedef std::vector< JacobianType >                 JacobianVectorType;
    typedef itk::Statistics::MersenneTwisterRandomVariateGenerator 
                                                        RandomGeneratorType;
    typedef typename JacobianType::ValueType            JacobianValueType;

    /** If the first step, and automatic gain estimation is desired,
     * then estimate the gain (Param_a) parameter automatically.
     * The following formula is used:
     * a = t^2 / max( E( z'B'Bz ) + 2 sqrt( Var( z'B'Bz ) ) )
     * b = ||g||^2 / N
     * Param_a = chosen such that gain at iteration 0 equals sqrt(a/b);
     * where:
     *   t = a user-defined parameter: the desired maximum voxel displacement in mm
     *   B = dT/dmu = the transform jacobian at a voxel x
     *   z = a random variable ~ N(0,I)
     *   max = over the fixed image domain, 10000 samples.
     *
     * \todo: this description is not correct anymore
     */
    if ( this->m_AutomaticGainEstimation && (this->GetCurrentIteration()==0) )
    {
      
      const unsigned int N = static_cast<unsigned int>( this->GetGradient().GetSize() );
      const double Nd = static_cast<double>( N );
      const double t = this->m_InitialStepSize;
      
      /** Get fixed image and region */
      typename FixedImageType::ConstPointer fixedImage = this->GetRegistration()->
        GetAsITKBaseType()->GetFixedImage();
      FixedImageRegionType fixedRegion = this->GetRegistration()->
        GetAsITKBaseType()->GetFixedImageRegion();
            
      /** Get transform and set current position */
      typename TransformType::Pointer transform = this->GetRegistration()->
        GetAsITKBaseType()->GetTransform();
      transform->SetParameters( this->GetCurrentPosition() );
      const unsigned int outdim = transform->GetOutputSpaceDimension();

      /** Setup random iterator on fixed image */
      FixedImageIteratorType iter( fixedImage, fixedRegion );
      unsigned long nrofsamples = 100;
      const double maxmem = 200e6;
      if ( outdim * N * nrofsamples * sizeof(JacobianValueType) > maxmem )
      {
        nrofsamples = maxmem / outdim / N / sizeof(JacobianValueType);
      }
      iter.SetNumberOfSamples( nrofsamples );
      iter.GoToBegin();

      FixedImagePointType point;
      double maxstep = 0.0;

      const ScalesType & scales = this->m_ScaledCostFunction->GetScales();
      JacobianVectorType jacvec(nrofsamples);
      unsigned int s = 0;
     
      /** Loop over image and compute jacobian. Save the jacobians in a vector. */
      while ( !iter.IsAtEnd() )
      {
           
        const FixedImageIndexType & index = iter.GetIndex();
        fixedImage->TransformIndexToPhysicalPoint( index, point );
        jacvec[s] = transform->GetJacobian( point );

        if ( this->GetUseScales() )
        {
          for (unsigned int p = 0; p < N; ++p)
          {
            jacvec[s].scale_column( p, 1.0/scales[p] );
          }
        }        
     
        ++iter;
        ++s;
      } // end while     

      double frofrojac = 0.0;
      maxstep = 0.0;
      const double n = static_cast<double>(nrofsamples);
      for ( unsigned int x = 0 ; x < nrofsamples; ++x)
      {
        double step = 0.0;
        double frofrojacjac = 0.0;
        const JacobianType & jacx = jacvec[x];
        frofrojac += vnl_math_sqr( jacx.frobenius_norm() ) / n; // needed later

        JacobianType jac4(outdim,outdim);
        jac4.Fill(0.0); // for variance
        for ( unsigned int y = 0 ; y < nrofsamples; ++y)
        {
          const JacobianType & jacy = jacvec[y];
          JacobianType jacjac(outdim,outdim);

          for( unsigned int dx = 0; dx < outdim; ++dx )
          {
            for( unsigned int dy = 0; dy < outdim; ++dy )
            {
              jacjac(dx,dy)=0.0;
              for (unsigned int p = 0; p < N; ++p)
              {
                jacjac[dx][dy] += jacx[dx][p] * jacy[dy][p];
              } // p
            } // dy
          } // dx

          /** expec*/
          frofrojacjac += vnl_math_sqr( jacjac.frobenius_norm() ) / n;

          for( unsigned int dx = 0; dx < outdim; ++dx )
          {
            ParametersType jacjacdx(jacjac[dx], outdim, false);
            for( unsigned int dy = 0; dy < outdim; ++dy )
            {              
              ParametersType jacjacdy(jacjac[dy], outdim, false);
              jac4[dx][dy] += dot_product(jacjacdx, jacjacdy);
            }
          }                           
          
        }  // y
        /** std */
        const double frojac4 = vcl_sqrt(2.0) * jac4.frobenius_norm() / n;
        step = frofrojacjac + 2.0 * frojac4;
        maxstep = vnl_math_max( maxstep, step);
      } // x
      frofrojac /= n;
      maxstep /= n;
      jacvec.clear();

      maxstep = vcl_sqrt( maxstep );    
      
      /** perturbation gain = a, such that mu = a*N(0,I), and E(deltaT^2) = t^2 */
      const double perturbationgain = t / maxstep;
      
      /** Number of gradients to estimate the average square magnitude 
       * N * nrofparams = 50
       * This gives a probability of ~1 that the average square magnitude
       * does not exceed twice the real expected square magnitude, or half.  */
      const unsigned int numberofgradients = static_cast<unsigned int>(
        vcl_ceil( 50.0 / Nd ) );

      /** One we have already */
      double gg = this->GetGradient().squared_magnitude();
      double maxg = this->GetGradient().inf_norm();
      
      /** Compute gg for some random parameters */  
      double dummyval = 0.0;
      DerivativeType gradient;
      typename RandomGeneratorType::Pointer randomgenerator = RandomGeneratorType::New();
      for ( unsigned int i = 1 ; i < numberofgradients; ++i)
      {
      	/** Select new spatial samples for the computation of the metric */
		    if ( this->GetNewSamplesEveryIteration() )
        {
			    this->SelectNewSamples();
		    }
                
        /** Generate a perturbation */
        ParametersType perturbation = this->GetScaledCurrentPosition();
        for (unsigned int p = 0; p < N; ++p)
        {
          perturbation[p] += perturbationgain * 
            randomgenerator->GetNormalVariate(0.0, 1.0);
        }

        /** Get derivative and magnitude */
        this->GetScaledValueAndDerivative( perturbation, dummyval, gradient );
        gg += gradient.squared_magnitude();
        maxg = vnl_math_max( maxg, gradient.inf_norm() );
      }
      gg /= numberofgradients;

      /** We would like this gain at the first iteration (time0): */
      //double gain = perturbationgain * vcl_sqrt(Nd) / vcl_sqrt(gg);
      //double gain = perturbationgain * 1.0 / maxg;
      double gain = perturbationgain * vcl_sqrt(frofrojac) / vcl_sqrt(gg);
      elxout << "perturbgain = " << perturbationgain << std::endl;
      elxout << "sqrtfrofrojac = " << vcl_sqrt(frofrojac) << std::endl;
      elxout << "sqrtgg = " << vcl_sqrt(gg) << std::endl;
      elxout << "n = " << n << std::endl;

      /** With a=1 we would get the following gain: */
      double time0 = 0.0;
      if ( this->GetUseCruzAcceleration() )
      {
        time0 = this->GetInitialTime();
      }      
      this->SetParam_a(1.0);
      const double tempgain = this->Compute_a( time0 );
      /** So we have to set Param_a to gain/tempgain */
      this->SetParam_a( gain/tempgain );

      /** Print to log file */
      elxout << "Estimated value for SP_a (gain): " << gain/tempgain << std::endl;
            
    } // end if automatic gain estimation

		this->Superclass1::AdvanceOneStep();

	} //end AdvanceOneStep
	

} // end namespace elastix

#endif // end #ifndef __elxAcceleratedGradientDescent_hxx

