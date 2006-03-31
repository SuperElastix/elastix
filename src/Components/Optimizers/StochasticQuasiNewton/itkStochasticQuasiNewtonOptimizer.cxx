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
		this->m_NormalizeInitialGradients = true;

		this->m_GainFactor = 1.0;
		this->m_BetaMax = 2.0;
		this->m_DetMax = 2.0;
		this->m_Decay_A = 50.0;
		this->m_Decay_alpha = 0.602;
		this->m_Diag = 0.0;
		this->m_UpdateFactor = 0.0;
		this->m_UseHessian = false;
		this->m_NumberOfGradientDescentIterations = 25;
		
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
		this->m_UpdateFactor = 0.0;
		this->m_UseHessian = false;
				
    /** Initialize the scaledCostFunction with the currently set scales */
    this->InitializeScales();

    /** Set the current position as the scaled initial position */
    this->SetCurrentPosition( this->GetInitialPosition() );
		//scaled?

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
		 		
		if  ( this->GetCurrentIteration() <= this->GetNumberOfInitializationSteps() )
		{
			this->ComputeInitialSearchDirection(gradient, searchDir);
		}
		else
		{

			if ( this->GetCurrentIteration() >= this->GetNumberOfGradientDescentIterations() )
			{
				this->m_UseHessian = true;
			}

			if (this->GetUseHessian() == false)
			{
				this->UpdateHessianMatrix();
				/** For debugging: */
				/**const HessianMatrixType & H = this->m_H;
				vnl_symmetric_eigensystem<double> eigsys(H);
				for (unsigned int i = 0; i < numberOfParameters; ++i)
				{
					std::cout << "D[" << i << "] = " << eigsys.D[i] << std::endl;
				}*/
			}

			/** Set the gain */
			const double k = static_cast<double>(
				this->GetCurrentIteration() - this->GetNumberOfInitializationSteps() );
			const double decay = vcl_pow(
				(this->m_Decay_A + k ) / (this->m_Decay_A + k + 1.0),
				this->m_Decay_alpha );
			gain *= decay;



			if ( this->GetUseHessian() )
			{
				searchDir.SetSize(numberOfParameters);
				searchDir.Fill(0.0);
				const HessianMatrixType & H = this->m_H;
				for (unsigned int i = 0 ; i< numberOfParameters; ++i)
				{
					double & sd_i = searchDir[i];
					for (unsigned int j = 0;  j< numberOfParameters; ++j)
					{
						sd_i -= H(i,j) * gradient[j];
					}
					sd_i *= gain;
					std::cout << H(i,i) << std::endl;
				}
			}
			else
			{
				searchDir = ( - gain * this->m_Diag ) * gradient;
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
			//std::cout << "Skipping update because of too small denominators." << std::endl;
		}
		else
		{
						
			bool still_valid = true;
			do 
			{
				update_factor1 += resolution;
				update_factor2 = update_factor1;
				update_factor3 = update_factor1;
				const double detfac1 = 1.0 - update_factor1;
				const double temp = 1.0 / update_factor1 - 1.0;
				const double detfac2 = 1.0 + update_factor2 * ( sBs/ys + ys/(yHy*temp) );
				const double detfac3 = 1.0 + update_factor3 * phi * ( 
					-1.0 + sBs*yHy/(ys*ys) +
					( 2.0*sBs - sBs*sBs*yHy/(ys*ys) - ys*ys/yHy ) /
					( ys/update_factor2 + sBs + ys*ys/(temp*yHy) )   );
				
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

				const double yHynieuw = yHy + update_factor1*(ys-yHy);
				const double sBsnieuw = sBs + factor_B1*ys*ys + factor_B2*sBs*sBs +
					factor_B3*2.0*ys*sBs;
				
				if ( update_factor > (1.0 - resolution) )
				{
					still_valid = false;
				}

				if ( (yHynieuw > (beta2*yHy)) || (yHynieuw < (beta1*yHy)) )
				{
					still_valid = false;
				}
				if ( (sBsnieuw > (beta2*sBs)) || (sBsnieuw < (beta1*sBs)) )
				{
					still_valid = false;
				}

				/**if ( (detfac1*detfac2*detfac3 > det2) || (detfac1*detfac2*detfac3 < det1) )
				{
					still_valid = false;
				}*/
						
			} while ( still_valid );

			double small_number3 = 1e-10;
			
			update_factor1 -= resolution * ( 1.0 );
			update_factor1 = vnl_math_min( update_factor1, 1.0 - small_number3 );
			update_factor1 *= gain*gain;
			update_factor1 = vnl_math_min( update_factor1, 1.0 - small_number3 );

			update_factor2 = update_factor1;
			update_factor3 = update_factor1;
			update_factor  = update_factor1;
	
		}
		
		/** Save for interested users */
    this->m_UpdateFactor = update_factor;
				
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
		double & diag = this->m_Diag;
		double & gain = this->m_GainFactor;
		double currentGain;
		double gradientNormalizer = 1.0;
		
		if ( this->GetNormalizeInitialGradients() )
		{
			gradientNormalizer = gradient.magnitude();
		}
		
		if ( this->GetCurrentIteration() == 0 )
		{
			gain = this->GetInitialStepLengthEstimate();
			currentGain = gain / gradientNormalizer;
			diag = currentGain;
		}
		else if (this->GetCurrentIteration() <= this->GetNumberOfInitializationSteps())
		{

			const double k = static_cast<double>(this->GetCurrentIteration());
			const double kplus1 = static_cast<double>(this->GetCurrentIteration()+1);
			const double small_number1 = 1e-10;
			const ParametersType & s = this->m_Step;
			const DerivativeType & y = this->m_GradientDifference;
			const double sy = inner_product( y, s );
			const double yy = y.squared_magnitude() + small_number1;
			double diag2 = sy/yy;
			
			/** Update the estimate for the initial diagonal hessian */
			diag2 = (k*diag + diag2) / kplus1;
			diag = vnl_math_max( 0.5 * diag, vnl_math_min( 2.0 * diag, diag2) );
			//we could also take the median instead....
	
			/** Compute the new gain */
			if (this->GetCurrentIteration() < this->GetNumberOfInitializationSteps())
			{
				const double isle = this->GetInitialStepLengthEstimate();
				gain = isle - k * ( isle - 1.0/isle ) /
					static_cast<double>( this->GetNumberOfInitializationSteps() - 1 );
				//of exponentieel? en/of user een range laten opgeven misschien?
				currentGain = gain / gradientNormalizer;
			}
			
		}

		if (this->GetCurrentIteration() == this->GetNumberOfInitializationSteps())
		{
				this->m_H.fill_diagonal( vcl_abs(diag) );
				this->m_B.fill_diagonal( 1.0 / vcl_abs(diag) );
				currentGain = vcl_abs(diag);
				
				/** reset the gain to 1.0 */
				gain = 1.0;
		}

		searchDir = ( - currentGain ) * gradient;

	} // end ComputeInitialSearchDirection



} // end namespace itk


#endif // #ifndef __itkStochasticQuasiNewtonOptimizer_cxx

