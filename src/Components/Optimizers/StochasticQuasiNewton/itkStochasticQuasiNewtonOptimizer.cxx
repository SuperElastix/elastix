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
		this->m_MinimumMemory = 2;
		this->m_InitialStepLengthEstimate = 1.0;

		this->m_ss_ys = 1.0;
		this->m_ys_yy = 0.0;
		this->m_NumberOfUpdates = 0;
		this->m_GainFactor = 1.0;
		
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
		this->m_S.SetSize(numberOfParameters, numberOfParameters);
		this->m_S.Fill(0.0);
		this->m_ss_ys = 1.0;
		this->m_ys_yy = 0.0;
		this->m_NumberOfUpdates = 0;
		this->m_GainFactor = 1.0;
		
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
			this->m_NumberOfUpdates++;
					  
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

		if (this->m_NumberOfUpdates > 0)
		{
			const double _1_k = 1.0 / static_cast<double>(this->m_NumberOfUpdates);
			const double kmin1_k = static_cast<double>(this->m_NumberOfUpdates - 1) /
				static_cast<double>(this->m_NumberOfUpdates);
		
			const double ys = inner_product( this->m_GradientDifference, this->m_Step );
			const double ss =  this->m_Step.squared_magnitude();
			
			if (this->m_NumberOfUpdates ==1)
			{
				const double yy=this->m_GradientDifference.squared_magnitude();
				this->m_H.fill_diagonal( vcl_abs(ys/yy) );
				//this->m_H.fill_diagonal(vcl_abs(this->m_Step.squared_magnitude()/ys));
			}
			vnl_vector<double> Hy = this->m_H * this->m_GradientDifference;

			const double yHy = inner_product(this->m_GradientDifference, Hy);
			const double HyHy = Hy.squared_magnitude();
			const double sHy= inner_product( this->m_Step, Hy );

							
			//HessianMatrixType Hupd;
			//HessianMatrixType temp2= outer_product(this->m_Step, this->m_Step) / ys;
			//HessianMatrixType temp(numberOfParameters,numberOfParameters);
			//temp.Fill(0.0);
			//temp.fill_diagonal(1.0);
			//temp -= (1.0/ys)*outer_product(this->m_GradientDifference, this->m_Step);
			//bfgs:				Hupd = ( temp.transpose() * (this->m_H * temp) + temp2) - this->m_H;

			//dfp:
			HessianMatrixType Hupd = outer_product(this->m_Step, this->m_Step) / ys -
			  ( outer_product(Hy,Hy)/ yHy );
		
			const double H_frob = this->m_H.frobenius_norm();
			const double Hupdate_frob = vcl_sqrt(
				ss*ss/(ys*ys)+ HyHy*HyHy/(yHy*yHy) - 2*sHy*sHy/(ys*yHy)  );
			
			const double frob_factor = vnl_math_min(1.0, H_frob * _1_k / Hupdate_frob);
			const double frob_ys = frob_factor / ys;
			const double frob_yHy = frob_factor / yHy;

			HessianMatrixType & H = this->m_H;
			const ParametersType & s = this->m_Step;
			for (unsigned int i = 0 ; i< numberOfParameters; ++i)
			{
				for (unsigned int j = 0;  j< numberOfParameters; ++j)
				{
					H(i,j)+= ( s[i]*s[j] / frob_ys - Hy[i]*Hy[j] / frob_yHy );
				} // end for j
			} // end for i

						
			//if (0)//( ys > 1e-14 ) //enforce a positive definite H.
			//{
			//	
			//	const double ss = this->m_Step.squared_magnitude();
			//	const double ss_ysk = ss / ys;
			//	
			//	const double _1_ss = 1.0 / ss;
			//	const double _1_ssk = _1_ss * _1_k;
			//	const double ss_ys_old = this->m_ss_ys;
			//	//this->m_ss_ys =	kmin1_k *	this->m_ss_ys + _1_k * ss_ysk;

			//	if (1)// (ss_ysk > this->m_ss_ys)
			//	{
			//		this->m_ss_ys =	1.0 / (kmin1_k / this->m_ss_ys + _1_k / ss_ysk);
			//	}
			//	else
			//	{
			//		this->m_ss_ys = kmin1_k * this->m_ss_ys + _1_k * ss_ysk;
			//	}
			//															
			//	const double deltac = this->m_ss_ys - ss_ys_old;

			//	const double alpha = ( ss_ysk - this->m_ss_ys ) * _1_ssk; 
			//	const double diag_update = _1_k * ss_ys_old + deltac;
			//	const double gamma = - deltac * kmin1_k;
			//	
			//	HessianMatrixType & H = this->m_H;
			//	HessianMatrixType & S = this->m_S;
			//	const ParametersType & s = this->m_Step;
			//
			//	for (unsigned int i = 0 ; i< numberOfParameters; ++i)
			//	{
			//		for (unsigned int j = 0;  j< numberOfParameters; ++j)
			//		{
			//			const double si_sj = s[i]*s[j];
			//			double & S_ij = S(i,j);
			//			double & H_ij = H(i,j);
			//			H_ij = kmin1_k * H_ij + alpha * si_sj + gamma * S_ij;
			//			S_ij = kmin1_k * S_ij + si_sj * _1_ssk;
			//		} // end for j
			//		H(i,i) += diag_update;
			//	} // end for i
			//	
			//} // end if ys> 1e-14
			//else
			//{
			//  this->m_H *= kmin1_k;			
			//}
			      				
		} // end if nr of updates >0
		    		
		if ( this->m_NumberOfUpdates < this->GetMinimumMemory() )
		{
			searchDir = ( - this->m_GainFactor * this->GetInitialStepLengthEstimate() /
	      gradient.magnitude() ) * gradient;
    }
		else // number of updates > minimummemory
		{
			/**vnl_symmetric_eigensystem<double> eigsys(this->m_H);
			for (unsigned int j = 0;  j< numberOfParameters; ++j)
			{
				std::cout << "D[" << j << "] = \t" << eigsys.D[j] << std::endl;
			}*/

			searchDir.SetSize(numberOfParameters);
			searchDir.Fill(0.0);
			const HessianMatrixType & H = this->m_H;
			const double & gain = this->m_GainFactor;
			for (unsigned int i = 0 ; i< numberOfParameters; ++i)
			{
				double & sd_i = searchDir[i];
				for (unsigned int j = 0;  j< numberOfParameters; ++j)
				{
					sd_i -= H(i,j) * gradient[j];
				} // end for j
				sd_i *= gain;
			} // end for i

		}  //end if number of updates is enough.

	} // end ComputeSearchDirection



} // end namespace itk


#endif // #ifndef __itkStochasticQuasiNewtonOptimizer_cxx

