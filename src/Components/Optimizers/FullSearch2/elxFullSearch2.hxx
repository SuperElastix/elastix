/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxFullSearch2_hxx
#define __elxFullSearch2_hxx

#include "elxFullSearch2.h"
#include <iomanip>
#include <string>
#include "math.h"

namespace elastix
{
using namespace itk;


  /**
   * ********************* Constructor ****************************
   */

  template <class TElastix>
    FullSearch2<TElastix>
    ::FullSearch2()
  {
  } // end Constructor


  /**
   * ***************** BeforeRegistration ***********************
   */

  template <class TElastix>
    void FullSearch2<TElastix>::
    BeforeRegistration(void)
  {
    /** Add some target cells to xout["iteration"].*/
    xout["iteration"].AddTargetCell("2:Metric");

    /** Format them as floats */
    xl::xout["iteration"]["2:Metric"]   << std::showpoint << std::fixed;

  } // end BeforeRegistration


  /**
   * ***************** BeforeEachResolution ***********************
   */

  template <class TElastix>
    void FullSearch2<TElastix>
    ::BeforeEachResolution(void)
  {
    /** Get the current resolution level.*/
  //  unsigned int level = static_cast<unsigned int>(
  //    this->m_Registration->GetAsITKBaseType()->GetCurrentLevel() );

    /** Set the steps. */
    double min = -0.2;
    double max = 0.2;
    double step = 0.05;

    this->GetConfiguration()->ReadParameter( min,  "Steps", 0 );
    this->GetConfiguration()->ReadParameter( max,  "Steps", 1 );
    this->GetConfiguration()->ReadParameter( step, "Steps", 2 );

    std::vector<double> steps( 0 );
    double current = min;
    while ( current <= max )
    {
      steps.push_back( current );
      current += step;
    }

    this->SetStep( steps );

    /** Plug in the base parameters. */
    this->SetBasePosition( this->GetElastix()->GetElxTransformBase()->GetInitialTransform()->GetParameters() );

    /** Add some target cells to xout["iteration"].*/
    xout["iteration"].AddTargetCell("1:step");

    /** Format them as floats */
    xl::xout["iteration"]["1:step"]   << std::showpoint << std::fixed;

  } // end BeforeEachResolution


  /**
   * ***************** AfterEachIteration *************************
   */

  template <class TElastix>
    void FullSearch2<TElastix>
    ::AfterEachIteration(void)
  {
    unsigned int i = this->GetCurrentIteration();

    /** Print some information */
    xl::xout["iteration"]["2:Metric"]   << this->GetValue();
    xl::xout["iteration"]["1:step"]   << this->GetStep( i );

    /** Select new spatial samples for the computation of the metric
     * \todo You may also choose to select new samples after evaluation
     * of the metric value *
    if ( this->GetNewSamplesEveryIteration() )
    {
      this->SelectNewSamples();
    }*/

  } // end AfterEachIteration


  /**
   * ***************** AfterEachResolution *************************
   */

  template <class TElastix>
    void FullSearch2<TElastix>
    ::AfterEachResolution(void)
  {
    std::string stopcondition;

    switch ( this->GetStopCondition() )
    {
      case FullRangeSearched :
        stopcondition = "The full range has been searched";
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

    /** Print the best metric value */
    elxout
      << std::endl
      << "Best metric value in this resolution = "
      << this->GetBestValue()
      << std::endl;

  } // end AfterEachResolution


  /**
   * ******************* AfterRegistration ************************
   */

  template <class TElastix>
    void FullSearch2<TElastix>
    ::AfterRegistration(void)
  {
    /** Print the best metric value */
    double bestValue = this->GetBestValue();
    elxout << std::endl << "Final metric value  = " << bestValue  << std::endl;

  } // end AfterRegistration


  /**
   * ******************* StartOptimization ***********************
   *

  template <class TElastix>
    void FullSearch2<TElastix>
    ::StartOptimization(void)
  {

    this->Superclass1::StartOptimization();

  } //end StartOptimization */


} // end namespace elastix

#endif // end #ifndef __elxFullSearchOptimizer2_hxx

