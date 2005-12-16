#ifndef __itkStochasticQuasiNewtonOptimizer_cxx
#define __itkStochasticQuasiNewtonOptimizer_cxx

#include "itkStochasticQuasiNewtonOptimizer.h"
#include "vnl/vnl_math.h"
#include "vnl/algo/vnl_symmetric_eigensystem.h"

namespace itk
{

  /**
   * ******************** Constructor *************************
   */

  StochasticQuasiNewtonOptimizer::StochasticQuasiNewtonOptimizer()
  {
    itkDebugMacro("Constructor");

    this->m_CurrentValue = NumericTraits<MeasureType>::Zero;
    this->m_CurrentIteration = 0;
    this->m_StopCondition = Unknown;
    this->m_Stop = false;
    this->m_CurrentStepLength = 1.0;
      
    this->m_MaximumNumberOfIterations = 100;
		this->m_NumberOfInitializationSteps = 5;
		this->m_InitialStepLengthEstimate = 2.0;

		this->m_GainFactor = 1.0;
		this->m_BetaMax = 2.0;
		this->m_DetMax = 2.0;
		this->m_Decay_A = 50.0;
		this->m_Decay_alpha = 0.602;
		this->m_Diag = 0.0;		
		
  } // end constructor


  /**
   * ******************* StartOptimization *********************
   */

  void
    StochasticQuasiNewtonOptimizer::
    StartOptimization()
  {
    itkDebugMacro("StartOptimization");

    /** Reset some variables */
    this->m_Stop = false;
    this->m_StopCondition = Unknown;
    this->m_CurrentIteration = 0;
    this->m_CurrentStepLength = 1.0;
    this->m_CurrentValue = NumericTraits<MeasureType>::Zero;
				    
    /** Get the number of parameters; checks also if a cost function has been set at all.
    * if not: an exception is thrown */
    const unsigned int numberOfParameters =
      this->GetScaledCostFunction()->GetNumberOfParameters();

    /** Set the current gradient to (0 0 0 ...) */
    this->m_CurrentGradient.SetSize(numberOfParameters);
    this->m_CurrentGradient.Fill( 0.0 );
    		
		/** Initialize H,S */
		this->m_H.SetSize(numberOfParameters, numberOfParameters);
		this->m_H.Fill(0.0);
		this->m_B.SetSize(numberOfParameters, numberOfParameters);
		this->m_B.Fill(0.0);
		this->m_GainFactor = 1.0;
		this->m_Diag = 0.0;
				
    /** Initialize the scaledCostFunction with the currently set scales */
    this->InitializeScales();

    /** Set the current position as the scaled initial position */
    this->SetCurrentPosition( this->GetInitialPosition() );

    if ( !this->m_Stop )
    {
      this->ResumeOptimization();
    }
    
  } // end StartOptimization


  /**
   * ******************* ResumeOptimization *********************
   */

  void
    StochasticQuasiNewtonOptimizer::
    ResumeOptimization()
  {
    itkDebugMacro("ResumeOptimization");

    this->m_Stop = false;
    this->m_StopCondition = Unknown;
    
    ParametersType searchDir;
		DerivativeType previousGradient;
		
    this->InvokeEvent( StartEvent() );

    try
	  {
	    this->GetScaledValueAndDerivative(
	      this->GetScaledCurrentPosition(), 
	      this->m_CurrentValue,
	      this->m_CurrentGradient );
	  }
	  catch ( ExceptionObject& err )
	  {
	    this->m_StopCondition = MetricError;
	    this->StopOptimization();
	    throw err;
	  }

    /** Start iterating */
    while ( !this->m_Stop )
    {
    
      /** Check if the maximum number of iterations has not been exceeded */
      if ( this->GetCurrentIteration() >= this->GetMaximumNumberOfIterations() )
      {
        this->m_StopCondition = MaximumNumberOfIterations;
        this->StopOptimization();
        break;
      }

      /** Compute the new search direction, using the current gradient */

			this->ComputeSearchDirection(this->GetCurrentGradient(), searchDir );

      if ( this->m_Stop )
      {
        break;
      }

      /** Store the current gradient */
      previousGradient = this->GetCurrentGradient();

			this->m_CurrentStepLength = this->GetGainFactor();
			this->m_ScaledCurrentPosition += searchDir;

      this->InvokeEvent( IterationEvent() );
			      
      if ( this->m_Stop )
      {
        break;
      }
		
    	try
		  {
		    this->GetScaledValueAndDerivative(
		      this->GetScaledCurrentPosition(), 
		      this->m_CurrentValue,
		      this->m_CurrentGradient );
		  }
		  catch ( ExceptionObject& err )
		  {
		    this->m_StopCondition = MetricError;
		    this->StopOptimization();
		    throw err;
		  }

			if ( this->m_Stop )
      {
        break;
      }

			/** Store s and y */
			this->m_Step = searchDir;
			this->m_GradientDifference = this->GetCurrentGradient() - previousGradient;
					  
      this->m_CurrentIteration++;

    } // end while !m_Stop


  } // end ResumeOptimization


  /** 
   * *********************** StopOptimization *****************************
   */

  void
    StochasticQuasiNewtonOptimizer::
    StopOptimization()
  {
    itkDebugMacro("StopOptimization");
    this->m_Stop = true;
    this->InvokeEvent( EndEvent() );
  } // end StopOptimization()


  /** 
   * *********************** ComputeSearchDirection ************************
   */

  void
    StochasticQuasiNewtonOptimizer::
    ComputeSearchDirection(
      const DerivativeType & gradient,
      ParametersType & searchDir)
  {
    itkDebugMacro("ComputeSearchDirection");
		    
    const unsigned int numberOfParameters = gradient.GetSize();
		double & gain = this->m_GainFactor;
		 		
		if ( this->GetCurrentIteration() ==0 )
		{

			gain = this->GetInitialStepLengthEstimate();
			searchDir = ( - gain / gradient.magnitude() ) * gradient;
			this->m_Diag = gain / gradient.magnitude();

    }
		else if  ( this->GetCurrentIteration() <= this->GetNumberOfInitializationSteps() )
		{
			
			this->ComputeInitialSearchDirection(gradient, searchDir);

		}
		else
		{

			this->UpdateHessianMatrix();

			/** Set the gain */
			const double k = static_cast<double>(
				this->GetCurrentIteration() - this->GetNumberOfInitializationSteps() );
			const double decay = vcl_pow(
				(this->m_Decay_A + k ) / (this->m_Decay_A + k + 1.0),
				this->m_Decay_alpha );
			gain *= decay;
						
			searchDir.SetSize(numberOfParameters);
			searchDir.Fill(0.0);
			const HessianMatrixType & H = this->m_H;

			/** For debugging: */
			/**vnl_symmetric_eigensystem<double> eigsys(H);
			for (unsigned int i = 0; i < numberOfParameters; ++i)
			{
				std::cout << "D[" << i << "] = " << eigsys.D[i] << std::endl;
			}*/

			for (unsigned int i = 0 ; i< numberOfParameters; ++i)
			{
				double & sd_i = searchDir[i];
				for (unsigned int j = 0;  j< numberOfParameters; ++j)
				{
					sd_i -= H(i,j) * gradient[j];
				}
				sd_i *= gain;
			}
			
		}  //end if current iteration > NumberOfInitializationSteps

	} // end ComputeSearchDirection


	/** 
   * *********************** UpdateHessianMatrix ************************
	 *
	 * This function assumes that an initial hessian estimate is available,
	 * and that m_Step and m_GradientDifference are valid.
   */

  void
    StochasticQuasiNewtonOptimizer::
    UpdateHessianMatrix(void)
  {
    itkDebugMacro("UpdateHessianMatrix");
   
		double & gain = this->m_GainFactor;

		HessianMatrixType & H = this->m_H;
		HessianMatrixType & B = this->m_B;
		const ParametersType & s = this->m_Step;
		const DerivativeType & y = this->m_GradientDifference;
		const unsigned int numberOfParameters = s.GetSize();

		const double ys = inner_product( y, s );
		const vnl_vector<double> Bs = B * s;
		const double sBs = inner_product( s, Bs );
		const vnl_vector<double> Hy = H * y;
		const double yHy = inner_product(y, Hy);
		const vnl_vector<double> v = (s/ys) - (Hy/yHy);

		double phi = 1.0 + gain;
																		
		double beta1 = 1.0 / (this->m_BetaMax);
		double beta2 = this->m_BetaMax;
		double det1 =  1.0 / (this->m_DetMax);
		double det2 = this->m_DetMax;
		
		//testje:
		/**const double yHHy = Hy.squared_magnitude();
		const double vv = v.squared_magnitude();
		const double ss = s.squared_magnitude();
		const double yHv = inner_product(Hy,v);
		const double yHs = inner_product(Hy,s);
		const double sv = inner_product(s,v);
		const double N = static_cast<double>(numberOfParameters);
		const double sBBs = Bs.squared_magnitude();
		const double yy = y.squared_magnitude();
		const double sBy = inner_product(y,Bs);
		const double frobH = H.frobenius_norm();
		const double frobB = B.frobenius_norm();
		const double frobHupdconst = vcl_sqrt( 
			yHHy*yHHy/(yHy*yHy) + ss*ss/(ys*ys) 
			+ vv*vv*phi*phi*yHy*yHy - 2.0*yHs*yHs/(yHy*ys) - 2.0*phi*yHv*yHv
			+ 2.0*sv*sv*phi*yHy/ys );*/
		
		double update_factor = 0.0;
		double update_factor1 = 0.0;
		double update_factor2 = 0.0;
		double update_factor3 = 0.0;
		const double resolution = 1e-4;		
    					
		const double small_number2 = 1e-10;
		if (  ( vcl_abs(ys)< small_number2 ) || ( vcl_abs(yHy)< small_number2 )
			|| ((1.0-beta1)*resolution < small_number2) ||
			((beta2-1.0)*resolution < small_number2) )
		{
			std::cout << "Skipping update because of too small denominators." << std::endl;
		}
		else
		{
			
			
			bool still_valid = true;
			do 
			{
				//update_factor1 += resolution * ( 1.0 - beta1 );
				update_factor1 += resolution * ( 1.0  );
				update_factor2 = update_factor1;
				update_factor3 = update_factor1;
				const double detfac1 = 1.0 - update_factor1;
				const double temp = 1.0 / update_factor1 - 1.0;
				const double detfac2 = 1.0 + update_factor2 * ( sBs/ys + ys/(yHy*temp) );
				const double detfac3 = 1.0 + update_factor3 * phi * ( 
					-1.0 + sBs*yHy/(ys*ys) +
					( 2.0*sBs - sBs*sBs*yHy/(ys*ys) - ys*ys/yHy ) /
					( ys/update_factor2 + sBs + ys*ys/(temp*yHy) )   );

				/**if ( detfac1 < beta1 )
				{
					still_valid = false;
				}
				if ( ( detfac2 < beta1 ) || ( detfac2 > beta2 ) ) 
				{
					still_valid = false;
				}
				if ( ( detfac3 < beta1 ) || ( detfac3 > beta2 ) ) 
				{
					still_valid = false;
				}*/

				if ( (detfac1*detfac2*detfac3 > det2) || (detfac1*detfac2*detfac3 < det1) )
				{
					still_valid = false;
				}

				/**const double D = detfac1;
				const double E = 1.0 + update_factor2*sBs/ys;
				const double F = 1.0 + update_factor3*phi*(sBs*yHy/(ys*ys) - 1.0);
        if ( D < beta1 )
				{
					still_valid = false;
				}
				if ( ( E < beta1 ) || ( E > beta2 ) ) 
				{
					still_valid = false;
				}
				if ( F > beta2  ) 
				{
					still_valid = false;
				}*/

				
				const double uf2 = update_factor2;
				const double temp_r = (ys + uf2*sBs) / 
					( yHy*temp*(ys + uf2*sBs) + uf2*ys*ys );
				const double temp_t = -1.0 / ( ys/uf2 + sBs + ys*ys/(temp*yHy) );
				const double temp_u = -1.0 / ( yHy*temp*(1.0/uf2 + sBs/ys) + ys );
				const double temp_p = 1.0/ys + temp_t*(sBs/ys - ys/yHy);
				const double temp_q = -1.0/yHy + temp_u*(sBs/ys - ys/yHy);

				const double factor_B1 = temp_r - update_factor3*phi*temp_q*temp_q*yHy/detfac3;
				const double factor_B2 = temp_t - update_factor3*phi*temp_p*temp_p*yHy/detfac3;
				const double factor_B3 = temp_u - update_factor3*phi*temp_p*temp_q*yHy/detfac3;
					
				/**const double frobHupd = update_factor1*frobHupdconst;
				const double frobBupd = vcl_sqrt(
					factor_B1*factor_B1*yy*yy +
					factor_B2*factor_B2*sBBs*sBBs +
					factor_B3*factor_B3*2.0*(sBBs*yy + sBs*sBy) +
					factor_B1*factor_B2*2.0*sBy*sBy +
					factor_B1*factor_B3*4.0*yy*sBy +
					factor_B2*factor_B3*4.0*sBBs*sBy );*/

				if ( update_factor > (1.0 - resolution) )
				{
					still_valid = false;
				}
				/**if ( frobHupd > (beta2*frobH / vcl_sqrt(N)) )
				{
					still_valid = false;
				}
				if ( frobBupd > (beta2*frobB / vcl_sqrt(N)) )
				{
					still_valid = false;
				}*/

				const double yHynieuw = yHy + update_factor1*(ys-yHy);
				const double sBsnieuw = sBs + factor_B1*ys*ys + factor_B2*sBs*sBs +
					factor_B3*2.0*ys*sBs;
				
				if ( (yHynieuw > (beta2*yHy)) || (yHynieuw < (beta1*yHy)) )
				{
					still_valid = false;
				}
				if ( (sBsnieuw > (beta2*sBs)) || (sBsnieuw < (beta1*sBs)) )
				{
					still_valid = false;
				}
							
						
			} while ( still_valid );

			double small_number3 = 1e-10;
			//update_factor1 -= resolution * ( 1.0 - beta1 );
			update_factor1 -= resolution * ( 1.0 );
			update_factor1 = vnl_math_min( update_factor1, 1.0 - small_number3 );
			std::cout << "UpdateFactorPreGain: " << update_factor1 << std::endl; 
			update_factor1 *= gain*gain;
			
			update_factor1 = vnl_math_min( update_factor1, 1.0 - small_number3 );
			update_factor2 = update_factor1;
			update_factor3 = update_factor1;
			update_factor  = update_factor1;
	
		}
		std::cout << "UpdateFactor:        " << update_factor << std::endl; 
		
		double small_number4 = 1e-10;
		if ( update_factor > small_number4 )
		{
	
			const double factor_H1 = -update_factor1 / yHy;
			const double factor_H2 = update_factor2 / ys;
			const double factor_H3 = update_factor3 * phi * yHy;

      const double temp = 1.0 / update_factor1 - 1.0;
			const double uf2 = update_factor2;
			const double detfac3 = 1.0 + update_factor3 * phi * ( 
				-1.0 + sBs*yHy/(ys*ys) +
				( 2.0*sBs - sBs*sBs*yHy/(ys*ys) - ys*ys/yHy ) /
				( ys/uf2 + sBs + ys*ys/(temp*yHy) )   );
			const double temp_r = (ys + uf2*sBs) / 
				( yHy*temp*(ys + uf2*sBs) + uf2*ys*ys );
			const double temp_t = -1.0 / ( ys/uf2 + sBs + ys*ys/(temp*yHy) );
			const double temp_u = -1.0 / ( yHy*temp*(1.0/uf2 + sBs/ys) + ys );
			const double temp_p = 1.0/ys + temp_t*(sBs/ys - ys/yHy);
			const double temp_q = -1.0/yHy + temp_u*(sBs/ys - ys/yHy);

			const double factor_B1 = temp_r - update_factor3*phi*temp_q*temp_q*yHy/detfac3;
			const double factor_B2 = temp_t - update_factor3*phi*temp_p*temp_p*yHy/detfac3;
			const double factor_B3 = temp_u - update_factor3*phi*temp_p*temp_q*yHy/detfac3;
						
			for (unsigned int i = 0 ; i< numberOfParameters; ++i)
			{
				for (unsigned int j = 0;  j< numberOfParameters; ++j)
				{
					H(i,j) += factor_H1 * Hy[i] * Hy[j] + factor_H2 * s[i] * s[j] + factor_H3 * v[i] * v[j];
					B(i,j) += factor_B1 * y[i] * y[j] + factor_B2 * Bs[i] * Bs[j] + factor_B3 * ( Bs[i] * y[j] + y[i] * Bs[j] );
				}
			} 

		} // end if updatefactor > smallnumber


	} // end UpdateHessianMatrix


  /** 
   * *********************** ComputeInitialSearchDirection ************************
   */

  void
    StochasticQuasiNewtonOptimizer::
    ComputeInitialSearchDirection(
      const DerivativeType & gradient,
      ParametersType & searchDir)
  {
    itkDebugMacro("ComputeInitialSearchDirection");
		
		/** Update the initial estimate for the hessian, m_Diag. */

    const unsigned int numberOfParameters = gradient.GetSize();
		
		const double k = static_cast<double>(this->GetCurrentIteration());
		const double kplus1 = static_cast<double>(this->GetCurrentIteration()+1);
		const double small_number1 = 1e-10;
		const ParametersType & s = this->m_Step;
		const DerivativeType & y = this->m_GradientDifference;
		const double sy = inner_product( y, s );
		const double yy = y.squared_magnitude() + small_number1;
		double & h = this->m_Diag;
								
		double h2= sy/yy;
		h2 = (k*h + h2) / kplus1;
		h = vnl_math_max( 0.5 * h, vnl_math_min( 2.0 * h, h2) );
		//we could also take the median instead....
		    
		double & gain = this->m_GainFactor;
				
		if (this->GetCurrentIteration() == this->GetNumberOfInitializationSteps())
		{
				this->m_H.fill_diagonal( vcl_abs(h) );
				this->m_B.fill_diagonal( 1.0 / vcl_abs(h) );
				
				searchDir = ( - vcl_abs(h) ) * gradient;

				/** reset the gain to 1.0 */
				gain = 1.0;
		}
		else
		{
			/** Take a step in the direction of the normalised gradient */
			const double isle = this->GetInitialStepLengthEstimate();
			gain = isle - k * ( isle - 1.0/isle ) /
				static_cast<double>( this->GetNumberOfInitializationSteps() - 1 );

			searchDir = ( - gain / gradient.magnitude() ) * gradient;
		}

	} // end ComputeInitialSearchDirection




} // end namespace itk


#endif // #ifndef __itkStochasticQuasiNewtonOptimizer_cxx

