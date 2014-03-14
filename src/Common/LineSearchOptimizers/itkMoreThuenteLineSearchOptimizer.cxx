/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkMoreThuenteLineSearchOptimizer_cxx
#define __itkMoreThuenteLineSearchOptimizer_cxx

#include "itkMoreThuenteLineSearchOptimizer.h"
#include "vcl_limits.h"
#include "vnl/vnl_math.h"

namespace itk
{

/**
 * ****************** Constructor *****************************
 */

MoreThuenteLineSearchOptimizer
::MoreThuenteLineSearchOptimizer()
{
  this->m_f                         = NumericTraits< MeasureType >::Zero;
  this->m_dg                        = 0.0;
  this->m_InitialDerivativeProvided = false;
  this->m_InitialValueProvided      = false;
  this->m_MaximumNumberOfIterations = 20;
  this->m_ValueTolerance            = 1e-4;
  this->m_GradientTolerance         = 0.9;
  this->m_IntervalTolerance         = std::numeric_limits< double >::epsilon();
  this->SetMinimumStepLength( 1e-20 );
  this->SetMaximumStepLength( 1e20 );

  this->InitializeLineSearch();

} // end Constructor


/**
 * *************** SetInitialValue *****************************
 */

void
MoreThuenteLineSearchOptimizer
::SetInitialValue( MeasureType value )
{
  this->m_InitialValueProvided = true;
  this->m_f                    = value;
  this->Modified();

} // end SetInitialValue()


/**
 * *************** SetInitialDerivative *****************************
 */

void
MoreThuenteLineSearchOptimizer
::SetInitialDerivative( const DerivativeType & derivative )
{
  this->m_InitialDerivativeProvided = true;
  this->m_g                         = derivative;
  this->Modified();

} // end SetInitialDerivative()


/**
 * *************** GetCurrentValueAndDerivative *********************
 */

void
MoreThuenteLineSearchOptimizer
::GetCurrentValueAndDerivative(
  MeasureType & value, DerivativeType & derivative ) const
{
  value      = m_f;
  derivative = m_g;

} // end GetCurrentValueAndDerivative()


/**
 * ************************ GetCurrentDerivative *********************
 */

void
MoreThuenteLineSearchOptimizer
::GetCurrentDerivative( DerivativeType & derivative ) const
{
  derivative = m_g;

} // end GetCurrentValueAndDerivative()


/**
 * ************************ GetCurrentValue *********************
 */

MoreThuenteLineSearchOptimizer::MeasureType
MoreThuenteLineSearchOptimizer
::GetCurrentValue( void ) const
{
  return m_f;

} // end GetCurrentValueAndDerivative()


/**
 * ************ GetCurrentDirectionalDerivative *********************
 */

double
MoreThuenteLineSearchOptimizer
::GetCurrentDirectionalDerivative( void ) const
{
  return m_dg;

} // end GetCurrentDirectionalDerivative()


/**
 * ************************** StartOptimization *********************
 */

void
MoreThuenteLineSearchOptimizer
::StartOptimization( void )
{
  this->CheckSettings();

  this->SetCurrentPosition( this->GetInitialPosition() );
  this->GetInitialValueAndDerivative();
  this->m_dg = this->DirectionalDerivative( this->m_g );

  this->InitializeLineSearch();

  this->InvokeEvent( StartEvent() );

  if(  this->m_dginit >= 0 )
  {
    this->m_StopCondition = AscentSearchDirection;
    this->StopOptimization();
  }

  while( !this->m_Stop )
  {

    this->UpdateIntervalMinimumAndMaximum();
    this->BoundStep( this->m_step );
    this->PrepareForUnusualTermination();
    this->SetCurrentStepLength( this->m_step );
    this->ComputeCurrentValueAndDerivative();
    this->m_dg = this->DirectionalDerivative( this->m_g );
    this->TestConvergence( this->m_Stop );
    this->InvokeEvent( IterationEvent() );
    if( this->m_Stop )
    {
      this->StopOptimization();
      break;
    }
    this->m_CurrentIteration++;
    this->ComputeNewStepAndInterval();
    this->ForceSufficientDecreaseInIntervalWidth();

  } // end while

} // end StartOptimization()


/**
 * ******************** StopOptimization******************************
 */

void
MoreThuenteLineSearchOptimizer
::StopOptimization( void )
{
  this->m_Stop = true;
  this->InvokeEvent( EndEvent() );

} // end StopOptimization()


/**
 * ******************** CheckSettings *********************************
 */

int
MoreThuenteLineSearchOptimizer
::CheckSettings( void )
{
  if( this->GetCostFunction() == 0 )
  {
    itkExceptionMacro( << "CostFunction has not been set!" );
  }

  const unsigned int numberOfParameters
    = this->GetCostFunction()->GetNumberOfParameters();

  if( this->GetInitialPosition().GetSize() != numberOfParameters )
  {
    itkExceptionMacro( << "InitialPosition has incorrect dimension!" );
  }

  if( this->GetLineSearchDirection().GetSize() != numberOfParameters )
  {
    itkExceptionMacro( << "LineSearchDirection has incorrect dimension!" );
  }

  if( this->GetMinimumStepLength() <= 0.0 )
  {
    itkExceptionMacro( << "MinimumStepLength must be higher than zero!" );
  }

  if( this->GetMinimumStepLength() > this->GetMaximumStepLength() )
  {
    itkExceptionMacro( << "MinimumStepLength must be smaller than MaximumStepLength!" );
  }

  if( this->GetGradientTolerance() < this->GetValueTolerance() )
  {
    itkExceptionMacro( << "GradientTolerance must be greater than ValueTolerance!" );
  }
  return 0;

}  // end CheckSettings()


/**
 * *************** GetInitialValueAndDerivative *********************
 */

void
MoreThuenteLineSearchOptimizer
::GetInitialValueAndDerivative( void )
{

  if( !( this->m_InitialValueProvided && this->m_InitialDerivativeProvided ) )
  {

    try
    {
      if( !this->m_InitialValueProvided && !this->m_InitialDerivativeProvided )
      {
        this->GetCostFunction()->GetValueAndDerivative(
          this->GetInitialPosition(),
          this->m_f,
          this->m_g );
      }
      else if( !this->m_InitialValueProvided )
      {
        this->m_f = this->GetCostFunction()->GetValue(
          this->GetInitialPosition() );
      }
      else if( !this->m_InitialDerivativeProvided )
      {
        this->GetCostFunction()->GetDerivative(
          this->GetInitialPosition(),
          this->m_g );
      }

    }
    catch( ExceptionObject & err )
    {
      this->m_StopCondition = MetricError;
      //this->StopOptimization(); //not here since no start event has been generated yet

      /** Any user provided initial values/derivatives may not be
       * valid anymore */
      this->m_InitialDerivativeProvided = false;
      this->m_InitialValueProvided      = false;

      throw err;
    }
  } // end if no initial value or derivative provided.

  /** Any user provided initial values/derivatives are not
   * valid anymore */

  this->m_InitialDerivativeProvided = false;
  this->m_InitialValueProvided      = false;

} // end GetInitialValueAndDerivative()


/**
 * ***************** InitializeLineSearch ******************************
 *
 * Set some member variables to their initial values.
 * Assumes m_f and m_dg have been set already.
 */

void
MoreThuenteLineSearchOptimizer
::InitializeLineSearch( void )
{
  this->m_Stop                                 = false;
  this->m_StopCondition                        = Unknown;
  this->m_CurrentIteration                     = 0;
  this->m_SufficientDecreaseConditionSatisfied = false;
  this->m_CurvatureConditionSatisfied          = false;
  this->m_CurrentStepLength                    = 0.0;

  this->m_finit                 = this->m_f;
  this->m_fx                    = this->m_finit;
  this->m_fy                    = this->m_finit;
  this->m_step                  = this->GetInitialStepLengthEstimate();
  this->m_stepx                 = 0.0;
  this->m_stepy                 = 0.0;
  this->m_stepmin               = 0.0;
  this->m_stepmax               = 0.0;
  this->m_dginit                = this->m_dg;
  this->m_dgx                   = this->m_dginit;
  this->m_dgy                   = this->m_dginit;
  this->m_dgtest                = this->GetValueTolerance() * this->m_dginit;
  this->m_width                 = this->GetMaximumStepLength() - this->GetMinimumStepLength();
  this->m_width1                = this->m_width / 0.5;
  this->m_brackt                = false;
  this->m_stage1                = true;
  this->m_SafeGuardedStepFailed = false;

} // end InitializeLineSearch()


/**
 * *************** UpdateIntervalMinimumAndMaximum ********************
 *
 * Set the minimum and maximum steps to correspond to the present
 * interval of uncertainty
 */

void
MoreThuenteLineSearchOptimizer
::UpdateIntervalMinimumAndMaximum( void )
{
  const double xtrapf = 4.0;

  if( this->m_brackt )
  {
    this->m_stepmin = vnl_math_min( this->m_stepx, this->m_stepy );
    this->m_stepmax = vnl_math_max( this->m_stepx, this->m_stepy );
  }
  else
  {
    this->m_stepmin = this->m_stepx;
    this->m_stepmax
      = this->m_step + xtrapf * ( this->m_step - this->m_stepx );
  }

} // end UpdateIntervalMinimumAndMaximum()


/**
 * ************************* BoundStep ********************************
 *
 * Force a step to be within the bounds MinimumStepLength and
 * MaximumStepLength
 */

void
MoreThuenteLineSearchOptimizer
::BoundStep( double & step ) const
{
  step = vnl_math_max( step, this->GetMinimumStepLength() );
  step = vnl_math_min( step, this->GetMaximumStepLength() );

} // end BoundStep()


/**
 * ******************** PrepareForUnusualTermination ********************
 *
 * If an unusual termination is to occur then let m_step be the lowest
 * point obtained so far
 */

void
MoreThuenteLineSearchOptimizer
::PrepareForUnusualTermination( void )
{
  if( ( this->m_brackt
    && ( this->m_step <= this->m_stepmin
    || this->m_step >= this->m_stepmax ) )
    || this->m_CurrentIteration >= this->GetMaximumNumberOfIterations() - 1
    || this->m_SafeGuardedStepFailed
    || ( this->m_brackt
    && this->m_stepmax - this->m_stepmin
    <= this->GetIntervalTolerance() * this->m_stepmax ) )
  {
    this->m_step = this->m_stepx;
  }

} // end PrepareForUnusualTermination()


/**
 * ***************** ComputeCurrentValueAndDerivative ********************
 *
 * Ask the cost function to compute m_f and m_g at the current position.
 */

void
MoreThuenteLineSearchOptimizer
::ComputeCurrentValueAndDerivative( void )
{
  try
  {
    this->GetCostFunction()->GetValueAndDerivative(
      this->GetCurrentPosition(), this->m_f, this->m_g );
  }
  catch( ExceptionObject & err )
  {
    this->m_StopCondition = MetricError;
    this->StopOptimization();
    throw err;
  }

} // end ComputeCurrentValueAndDerivative()


/**
 * ************************** TestConvergence ****************************
 *
 * Test for convergence.
 */

void
MoreThuenteLineSearchOptimizer
::TestConvergence( bool & stop )
{
  stop = false;
  const double & step = this->m_step;

  MeasureType ftest1 = this->m_finit + step * this->m_dgtest;

  this->m_SufficientDecreaseConditionSatisfied = ( this->m_f <= ftest1 );
  this->m_CurvatureConditionSatisfied          = ( vnl_math_abs( this->m_dg ) <=
    this->GetGradientTolerance() * ( -this->m_dginit ) );

  if( ( this->m_brackt
    && ( step <= this->m_stepmin || step >= this->m_stepmax ) )
    || this->m_SafeGuardedStepFailed )
  {
    this->m_StopCondition = RoundingError;
    stop                  = true;
  }

  if( step == this->GetMaximumStepLength()
    && this->m_SufficientDecreaseConditionSatisfied
    && this->m_dg <= this->m_dgtest )
  {
    this->m_StopCondition = StepTooLarge;
    stop                  = true;
  }

  if( step == this->GetMinimumStepLength()
    && ( !( this->m_SufficientDecreaseConditionSatisfied )
    || this->m_dg >= this->m_dgtest ) )
  {
    this->m_StopCondition = StepTooSmall;
    stop                  = true;
  }

  if( this->m_CurrentIteration >= this->GetMaximumNumberOfIterations() - 1 )
  {
    this->m_StopCondition = MaximumNumberOfIterations;
    stop                  = true;
  }

  if( this->m_brackt
    && this->m_stepmax - this->m_stepmin
    <= this->GetIntervalTolerance() * this->m_stepmax )
  {
    this->m_StopCondition = IntervalTooSmall;
    stop                  = true;
  }

  if( this->m_SufficientDecreaseConditionSatisfied
    && this->m_CurvatureConditionSatisfied  )
  {
    this->m_StopCondition = StrongWolfeConditionsSatisfied;
    stop                  = true;
  }

} // end TestConvergence()


/**
 * ****************** ComputeNewStepAndInterval ************************
 *
 * Update the interval of uncertainty and compute the new step
 */

void
MoreThuenteLineSearchOptimizer
::ComputeNewStepAndInterval( void )
{
  int returncode = 0;

  /** In the first stage we seek a step for which the modified
  * function has a nonpositive value and nonnegative derivative. */

  /** Stage1? */
  if( this->m_stage1
    && this->m_SufficientDecreaseConditionSatisfied
    && this->m_dg >= this->m_dginit * vnl_math_min(
    this->GetValueTolerance(), this->GetGradientTolerance() ) )
  {
    this->m_stage1 = false;
  }

  /* A modified function is used to predict the step only if
  * we have not obtained a step for which the modified
  * function has a nonpositive function value and nonnegative
  * derivative, and if a lower function value has been
  * obtained but the decrease is not sufficient. */

  if( this->m_stage1
    && this->m_f <= this->m_fx
    && !( this->m_SufficientDecreaseConditionSatisfied ) )
  {
    /* Define the modified function and derivative values. */
    const double & dgtest = this->m_dgtest;

    double fm   = this->m_f - this->m_step * dgtest;
    double fxm  = this->m_fx - this->m_stepx * dgtest;
    double fym  = this->m_fy - this->m_stepy * dgtest;
    double dgm  = this->m_dg - dgtest;
    double dgxm = this->m_dgx - dgtest;
    double dgym = this->m_dgy - dgtest;

    /* Call SafeGuardedStep to update the interval of uncertainty */
    /* and to compute the new step. */
    returncode = this->SafeGuardedStep(
      this->m_stepx, fxm, dgxm,
      this->m_stepy, fym, dgym,
      this->m_step, fm, dgm,
      this->m_brackt,
      this->m_stepmin, this->m_stepmax );

    /* Reset the function and gradient values. */
    this->m_fx  = fxm + this->m_stepx * dgtest;
    this->m_fy  = fym + this->m_stepy * dgtest;
    this->m_dgx = dgxm + dgtest;
    this->m_dgy = dgym + dgtest;
  }
  else
  {
    /* Call SafeGuardedStep to update the interval of uncertainty */
    /* and to compute the new step. */
    returncode = this->SafeGuardedStep(
      this->m_stepx, this->m_fx, this->m_dgx,
      this->m_stepy, this->m_fy, this->m_dgy,
      this->m_step, this->m_f, this->m_dg,
      this->m_brackt,
      this->m_stepmin, this->m_stepmax );
  }

  if( returncode == 0 )
  {
    this->m_SafeGuardedStepFailed = true;
  }

} //end ComputeNewStepAndInterval()


/**
 * ************** ForceSufficientDecreaseInIntervalWidth ******************
 *
 * Force a sufficient decrease in the size of the interval of uncertainty
 */

void
MoreThuenteLineSearchOptimizer
::ForceSufficientDecreaseInIntervalWidth( void )
{
  if( this->m_brackt )
  {
    const double & stx = this->m_stepx;
    const double & sty = this->m_stepy;

    if( vnl_math_abs( sty - stx ) >= .66 * this->m_width1 )
    {
      this->m_step = stx + .5 * ( sty - stx );
    }
    this->m_width1 = this->m_width;
    this->m_width  = vnl_math_abs( sty - stx );
  }

} // end ForceSufficientDecreaseInIntervalWidth()


/**
 * ************************** SafeGuardedStep ****************************
 *
 * Advance a step along the line search direction and update
 * the interval of uncertainty. Returns 0 if an error has occurred.
 */

int
MoreThuenteLineSearchOptimizer
::SafeGuardedStep(
  double & stx, double & fx, double & dx,
  double & sty, double & fy, double & dy,
  double & stp, const double & fp, const double & dp,
  bool & brackt,
  const double & stpmin, const double & stpmax ) const
{
  /** This function is largely just a copy of the following function
  * taken from netlib/lbfgs.c */

  /** Original documentation: */

  /**    void mcstep_(stx, fx, dx, sty, fy, dy, stp, fp, dp,
  *             brackt, stpmin, stpmax, info)
  *      doublereal *stx, *fx, *dx, *sty, *fy, *dy, *stp, *fp, *dp;
  *      logical *brackt;
  *      doublereal *stpmin, *stpmax;
  *      integer *info;
  */

  /*     SUBROUTINE MCSTEP */

  /*     THE PURPOSE OF MCSTEP IS TO COMPUTE A SAFEGUARDED STEP FOR */
  /*     A LINESEARCH AND TO UPDATE AN INTERVAL OF UNCERTAINTY FOR */
  /*     A MINIMIZER OF THE FUNCTION. */

  /*     THE PARAMETER STX CONTAINS THE STEP WITH THE LEAST FUNCTION */
  /*     VALUE. THE PARAMETER STP CONTAINS THE CURRENT STEP. IT IS */
  /*     ASSUMED THAT THE DERIVATIVE AT STX IS NEGATIVE IN THE */
  /*     DIRECTION OF THE STEP. IF BRACKT IS SET TRUE THEN A */
  /*     MINIMIZER HAS BEEN BRACKETED IN AN INTERVAL OF UNCERTAINTY */
  /*     WITH ENDPOINTS STX AND STY. */

  /*     THE SUBROUTINE STATEMENT IS */

  /*       SUBROUTINE MCSTEP(STX,FX,DX,STY,FY,DY,STP,FP,DP,BRACKT, */
  /*                        STPMIN,STPMAX,INFO) */

  /*     WHERE */

  /*       STX, FX, AND DX ARE VARIABLES WHICH SPECIFY THE STEP, */
  /*         THE FUNCTION, AND THE DERIVATIVE AT THE BEST STEP OBTAINED */
  /*         SO FAR. THE DERIVATIVE MUST BE NEGATIVE IN THE DIRECTION */
  /*         OF THE STEP, THAT IS, DX AND STP-STX MUST HAVE OPPOSITE */
  /*         SIGNS. ON OUTPUT THESE PARAMETERS ARE UPDATED APPROPRIATELY. */

  /*       STY, FY, AND DY ARE VARIABLES WHICH SPECIFY THE STEP, */
  /*         THE FUNCTION, AND THE DERIVATIVE AT THE OTHER ENDPOINT OF */
  /*         THE INTERVAL OF UNCERTAINTY. ON OUTPUT THESE PARAMETERS ARE */
  /*         UPDATED APPROPRIATELY. */

  /*       STP, FP, AND DP ARE VARIABLES WHICH SPECIFY THE STEP, */
  /*         THE FUNCTION, AND THE DERIVATIVE AT THE CURRENT STEP. */
  /*         IF BRACKT IS SET TRUE THEN ON INPUT STP MUST BE */
  /*         BETWEEN STX AND STY. ON OUTPUT STP IS SET TO THE NEW STEP. */

  /*       BRACKT IS A LOGICAL VARIABLE WHICH SPECIFIES IF A MINIMIZER */
  /*         HAS BEEN BRACKETED. IF THE MINIMIZER HAS NOT BEEN BRACKETED */
  /*         THEN ON INPUT BRACKT MUST BE SET FALSE. IF THE MINIMIZER */
  /*         IS BRACKETED THEN ON OUTPUT BRACKT IS SET TRUE. */

  /*       STPMIN AND STPMAX ARE INPUT VARIABLES WHICH SPECIFY LOWER */
  /*         AND UPPER BOUNDS FOR THE STEP. */

  /*       INFO IS AN INTEGER OUTPUT VARIABLE SET AS FOLLOWS: */
  /*         IF INFO = 1,2,3,4,5, THEN THE STEP HAS BEEN COMPUTED */
  /*         ACCORDING TO ONE OF THE FIVE CASES BELOW. OTHERWISE */
  /*         INFO = 0, AND THIS INDICATES IMPROPER INPUT PARAMETERS. */

  /*     SUBPROGRAMS CALLED */

  /*       FORTRAN-SUPPLIED ... ABS,MAX,MIN,SQRT */

  /*     ARGONNE NATIONAL LABORATORY. MINPACK PROJECT. JUNE 1983 */
  /*     JORGE J. MORE', DAVID J. THUENTE */

  /** The info variable is now replaced by the returncode. */
  /** The variables are no longer passed as pointers, but by reference. */

  /* System generated locals */
  double d__1;

  /* Local variables */
  double sgnd, stpc, stpf, stpq, p, q, gamma, r, s, theta;
  bool   bound;

  int returncode = 0;

  /* CHECK THE INPUT PARAMETERS FOR ERRORS. */

  if( ( brackt && ( stp <= vnl_math_min( stx, sty )
    || stp >= vnl_math_max( stx, sty ) ) )
    || dx * ( stp - stx ) >= 0.
    || stpmax < stpmin )
  {
    return returncode;
  }

  /* DETERMINE IF THE DERIVATIVES HAVE OPPOSITE SIGN. */

  sgnd = dp * ( dx / vnl_math_abs( dx ) );

  /*     FIRST CASE. A HIGHER FUNCTION VALUE. */
  /*     THE MINIMUM IS BRACKETED. IF THE CUBIC STEP IS CLOSER */
  /*     TO STX THAN THE QUADRATIC STEP, THE CUBIC STEP IS TAKEN, */
  /*     ELSE THE AVERAGE OF THE CUBIC AND QUADRATIC STEPS IS TAKEN. */

  if( fp > fx )
  {
    returncode = 1;
    bound      = true;
    theta      = ( fx - fp ) * 3 / ( stp - stx ) + dx + dp;
    s          = vnl_math_max(
      vnl_math_max( vnl_math_abs( theta ), vnl_math_abs( dx ) ),
      vnl_math_abs( dp ) );
    d__1  = theta / s;
    gamma = s * vcl_sqrt( d__1 * d__1 - dx / s * ( dp / s ) );
    if( stp < stx )
    {
      gamma = -gamma;
    }
    p    = gamma - dx + theta;
    q    = gamma - dx + gamma + dp;
    r    = p / q;
    stpc = stx + r * ( stp - stx );
    stpq = stx
      + dx / ( ( fx - fp ) / ( stp - stx ) + dx ) / 2 * ( stp - stx );
    if( vnl_math_abs( stpc - stx ) < vnl_math_abs( stpq - stx ) )
    {
      stpf = stpc;
    }
    else
    {
      stpf = stpc + ( stpq - stpc ) / 2;
    }
    brackt = true;

    /*     SECOND CASE. A LOWER FUNCTION VALUE AND DERIVATIVES OF */
    /*     OPPOSITE SIGN. THE MINIMUM IS BRACKETED. IF THE CUBIC */
    /*     STEP IS CLOSER TO STX THAN THE QUADRATIC (SECANT) STEP, */
    /*     THE CUBIC STEP IS TAKEN, ELSE THE QUADRATIC STEP IS TAKEN. */

  }
  else if( sgnd < 0. )
  {
    returncode = 2;
    bound      = false;
    theta      = ( fx - fp ) * 3 / ( stp - stx ) + dx + dp;
    s          = vnl_math_max(
      vnl_math_max( vnl_math_abs( theta ), vnl_math_abs( dx ) ),
      vnl_math_abs( dp ) );
    d__1  = theta / s;
    gamma = s * vcl_sqrt( d__1 * d__1 - dx / s * ( dp / s ) );
    if( stp > stx )
    {
      gamma = -gamma;
    }
    p    = gamma - dp + theta;
    q    = gamma - dp + gamma + dx;
    r    = p / q;
    stpc = stp + r * ( stx - stp );
    stpq = stp + dp / ( dp - dx ) * ( stx - stp );
    if( vnl_math_abs( stpc - stp ) > vnl_math_abs( stpq - stp ) )
    {
      stpf = stpc;
    }
    else
    {
      stpf = stpq;
    }
    brackt = true;

    /*     THIRD CASE. A LOWER FUNCTION VALUE, DERIVATIVES OF THE */
    /*     SAME SIGN, AND THE MAGNITUDE OF THE DERIVATIVE DECREASES. */
    /*     THE CUBIC STEP IS ONLY USED IF THE CUBIC TENDS TO INFINITY */
    /*     IN THE DIRECTION OF THE STEP OR IF THE MINIMUM OF THE CUBIC */
    /*     IS BEYOND STP. OTHERWISE THE CUBIC STEP IS DEFINED TO BE */
    /*     EITHER STPMIN OR STPMAX. THE QUADRATIC (SECANT) STEP IS ALSO */
    /*     COMPUTED AND IF THE MINIMUM IS BRACKETED THEN THE THE STEP */
    /*     CLOSEST TO STX IS TAKEN, ELSE THE STEP FARTHEST AWAY IS TAKEN. */

  }
  else if( vnl_math_abs( dp ) < vnl_math_abs( dx ) )
  {
    returncode = 3;
    bound      = true;
    theta      = ( fx - fp ) * 3 / ( stp - stx ) + dx + dp;
    s          = vnl_math_max(
      vnl_math_max( vnl_math_abs( theta ), vnl_math_abs( dx ) ),
      vnl_math_abs( dp ) );

    /* THE CASE GAMMA = 0 ONLY ARISES IF THE CUBIC DOES NOT TEND */
    /* TO INFINITY IN THE DIRECTION OF THE STEP. */

    d__1  = theta / s;
    d__1  = d__1 * d__1 - dx / s * ( dp / s );
    gamma = s * vcl_sqrt( vnl_math_max( 0., d__1 ) );
    if( stp > stx )
    {
      gamma = -gamma;
    }
    p = gamma - dp + theta;
    q = gamma + ( dx - dp ) + gamma;
    r = p / q;
    if( r < 0. && gamma != 0. )
    {
      stpc = stp + r * ( stx - stp );
    }
    else if( stp > stx )
    {
      stpc = stpmax;
    }
    else
    {
      stpc = stpmin;
    }
    stpq = stp + dp / ( dp - dx ) * ( stx - stp );
    if( brackt )
    {
      if( vnl_math_abs( stp - stpc ) < vnl_math_abs( stp - stpq ) )
      {
        stpf = stpc;
      }
      else
      {
        stpf = stpq;
      }
    }
    else
    {
      if( vnl_math_abs( stp - stpc ) > vnl_math_abs( stp - stpq ) )
      {
        stpf = stpc;
      }
      else
      {
        stpf = stpq;
      }
    }

    /*     FOURTH CASE. A LOWER FUNCTION VALUE, DERIVATIVES OF THE */
    /*     SAME SIGN, AND THE MAGNITUDE OF THE DERIVATIVE DOES */
    /*     NOT DECREASE. IF THE MINIMUM IS NOT BRACKETED, THE STEP */
    /*     IS EITHER STPMIN OR STPMAX, ELSE THE CUBIC STEP IS TAKEN. */

  }
  else
  {
    returncode = 4;
    bound      = false;
    if( brackt )
    {
      theta = ( fp - fy ) * 3 / ( sty - stp ) + dy + dp;
      s     = vnl_math_max(
        vnl_math_max( vnl_math_abs( theta ), vnl_math_abs( dy ) ),
        vnl_math_abs( dp ) );
      d__1  = theta / s;
      gamma = s * vcl_sqrt( d__1 * d__1 - dy / s * ( dp / s ) );
      if( stp > sty )
      {
        gamma = -gamma;
      }
      p    = gamma - dp + theta;
      q    = gamma - dp + gamma + dy;
      r    = p / q;
      stpc = stp + r * ( sty - stp );
      stpf = stpc;
    }
    else if( stp > stx )
    {
      stpf = stpmax;
    }
    else
    {
      stpf = stpmin;
    }
  }

  /*     UPDATE THE INTERVAL OF UNCERTAINTY. THIS UPDATE DOES NOT */
  /*     DEPEND ON THE NEW STEP OR THE CASE ANALYSIS ABOVE. */

  if( fp > fx )
  {
    sty = stp;
    fy  = fp;
    dy  = dp;
  }
  else
  {
    if( sgnd < 0. )
    {
      sty = stx;
      fy  = fx;
      dy  = dx;
    }
    stx = stp;
    fx  = fp;
    dx  = dp;
  }

  /*     COMPUTE THE NEW STEP AND SAFEGUARD IT. */

  stpf = vnl_math_min( stpmax, stpf );
  stpf = vnl_math_max( stpmin, stpf );
  stp  = stpf;
  if( brackt && bound )
  {
    if( sty > stx )
    {
      d__1 = stx + ( sty - stx ) * .66f;
      stp  = vnl_math_min( d__1, stp );
    }
    else
    {
      d__1 = stx + ( sty - stx ) * .66f;
      stp  = vnl_math_max( d__1, stp );
    }
  }

  return returncode;

} // end SafeGuardedStep()


/**
 * ****************** PrintSelf *****************************
 */

void
MoreThuenteLineSearchOptimizer
::PrintSelf( std::ostream & os, Indent indent ) const
{
  /** Call the superclass' PrintSelf. */
  Superclass::PrintSelf( os, indent );

  os << indent << "m_CurrentIteration: "
     << this->m_CurrentIteration << std::endl;
  os << indent << "m_InitialDerivativeProvided: "
     << ( this->m_InitialDerivativeProvided ? "true" : "false" ) << std::endl;
  os << indent << "m_InitialValueProvided: "
     << ( this->m_InitialValueProvided ? "true" : "false" ) << std::endl;
  os << indent << "m_StopCondition: "
     << this->m_StopCondition << std::endl;
  os << indent << "m_Stop: "
     << ( this->m_Stop ? "true" : "false" ) << std::endl;
  os << indent << "m_SufficientDecreaseConditionSatisfied: "
     << ( this->m_SufficientDecreaseConditionSatisfied ? "true" : "false" ) << std::endl;
  os << indent << "m_CurvatureConditionSatisfied: "
     << ( this->m_CurvatureConditionSatisfied ? "true" : "false" ) << std::endl;

  os << indent << "m_step: "
     << this->m_step << std::endl;
  os << indent << "m_stepx: "
     << this->m_stepx << std::endl;
  os << indent << "m_stepy: "
     << this->m_stepy << std::endl;
  os << indent << "m_stepmin: "
     << this->m_stepmin << std::endl;
  os << indent << "m_stepmax: "
     << this->m_stepmax << std::endl;

  os << indent << "m_f: "
     << this->m_f << std::endl;
  os << indent << "m_fx: "
     << this->m_fx << std::endl;
  os << indent << "m_fy: "
     << this->m_fy << std::endl;
  os << indent << "m_finit: "
     << this->m_finit << std::endl;

  os << indent << "m_g: "
     << this->m_g << std::endl;
  os << indent << "m_dg: "
     << this->m_dg << std::endl;
  os << indent << "m_dginit: "
     << this->m_dginit << std::endl;
  os << indent << "m_dgx: "
     << this->m_dgx << std::endl;
  os << indent << "m_dgy: "
     << this->m_dgy << std::endl;
  os << indent << "m_dgtest: "
     << this->m_dgtest << std::endl;

  os << indent << "m_width: "
     << this->m_width << std::endl;
  os << indent << "m_width1: "
     << this->m_width1 << std::endl;
  os << indent << "m_brackt: "
     << ( this->m_brackt ? "true" : "false" ) << std::endl;
  os << indent << "m_stage1: "
     << ( this->m_stage1 ? "true" : "false" ) << std::endl;
  os << indent << "m_SafeGuardedStepFailed: "
     << ( this->m_SafeGuardedStepFailed ? "true" : "false" ) << std::endl;

  os << indent << "m_MaximumNumberOfIterations: "
     << this->m_MaximumNumberOfIterations << std::endl;
  os << indent << "m_ValueTolerance: "
     << this->m_ValueTolerance << std::endl;
  os << indent << "m_GradientTolerance: "
     << this->m_GradientTolerance << std::endl;
  os << indent << "m_IntervalTolerance: "
     << this->m_IntervalTolerance << std::endl;

} // end PrintSelf()


} // end namespace itk

#endif // #ifndef __itkMoreThuenteLineSearchOptimizer_cxx
