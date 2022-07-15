/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef elxFullSearchOptimizer_hxx
#define elxFullSearchOptimizer_hxx

#include "elxFullSearchOptimizer.h"
#include <iomanip>
#include <sstream>
#include <string>
#include <vnl/vnl_math.h>

namespace elastix
{

/**
 * ********************* Constructor ****************************
 */

template <class TElastix>
FullSearch<TElastix>::FullSearch()
{
  this->m_OptimizationSurface = nullptr;

} // end Constructor


/**
 * ***************** BeforeRegistration ***********************
 */

template <class TElastix>
void
FullSearch<TElastix>::BeforeRegistration()
{
  /** Add the target cells "ItNr" and "Metric" to IterationInfo. */
  this->AddTargetCellToIterationInfo("2:Metric");

  /** Format the metric as floats. */
  this->GetIterationInfoAt("2:Metric") << std::showpoint << std::fixed;

} // end BeforeRegistration


/**
 * ***************** BeforeEachResolution ***********************
 */

template <class TElastix>
void
FullSearch<TElastix>::BeforeEachResolution()
{
  /** Get the current resolution level.*/
  unsigned int level = static_cast<unsigned int>(this->m_Registration->GetAsITKBaseType()->GetCurrentLevel());

  /** Read FullSearchRange from the parameter file. */

  /** declare variables */
  std::string        name("");
  unsigned int       param_nr = 0;
  double             minimum = 0;
  double             maximum = 0;
  double             stepsize = 0;
  const std::string  prefix = "FS";
  unsigned int       entry_nr = 0;
  unsigned int       nrOfSearchSpaceDimensions = 0;
  bool               found = true;
  bool               realGood = true;
  std::ostringstream makeString;

  /** Create fullFieldName, which is "FullSearchSpace0" at level 0. */
  makeString << "FullSearchSpace" << level;
  std::string fullFieldName = makeString.str();

  /** Loop as long as search ranges are defined. */
  while (found)
  {
    /** Try to read (silently) from the parameter file. */
    /** \todo check earlier, in BeforeAll, if the searchspace has been defined. */

    if (realGood && found)
    {
      found = this->GetConfiguration()->ReadParameter(name, fullFieldName, entry_nr, false);
      realGood = this->CheckSearchSpaceRangeDefinition(fullFieldName, found, entry_nr);
      ++entry_nr;
    }
    if (realGood && found)
    {
      found = this->GetConfiguration()->ReadParameter(param_nr, fullFieldName, entry_nr, false);
      realGood = this->CheckSearchSpaceRangeDefinition(fullFieldName, found, entry_nr);
      ++entry_nr;
    }
    if (realGood && found)
    {
      found = this->GetConfiguration()->ReadParameter(minimum, fullFieldName, entry_nr, false);
      realGood = this->CheckSearchSpaceRangeDefinition(fullFieldName, found, entry_nr);
      ++entry_nr;
    }
    if (realGood && found)
    {
      found = this->GetConfiguration()->ReadParameter(maximum, fullFieldName, entry_nr, false);
      realGood = this->CheckSearchSpaceRangeDefinition(fullFieldName, found, entry_nr);
      ++entry_nr;
    }
    if (realGood && found)
    {
      found = this->GetConfiguration()->ReadParameter(stepsize, fullFieldName, entry_nr, false);
      realGood = this->CheckSearchSpaceRangeDefinition(fullFieldName, found, entry_nr);
      ++entry_nr;
    }

    /** Setup this search range. */
    if (realGood && found)
    {
      /** Setup the Superclass */
      this->AddSearchDimension(param_nr, minimum, maximum, stepsize);

      /** Create name of this dimension. */
      makeString.str("");
      makeString << prefix << ((entry_nr / 5) - 1) << ":" << name << ":" << param_nr;

      /** Store the name and create a column in IterationInfo. */
      this->m_SearchSpaceDimensionNames[param_nr] = makeString.str();
      this->AddTargetCellToIterationInfo(makeString.str().c_str());

      /** Format this xout iteration column as float. */
      this->GetIterationInfoAt(makeString.str().c_str()) << std::showpoint << std::fixed;
    }
  } // end while

  if (realGood)
  {
    /** The number of dimensions. */
    nrOfSearchSpaceDimensions = this->GetNumberOfSearchSpaceDimensions();

    /** Create the image that will store the results of the full search. */
    this->m_OptimizationSurface = NDImageType::NewNDImage(nrOfSearchSpaceDimensions);
    this->m_OptimizationSurface->CreateNewImage();
    /** \todo don't do this if more than max allowable dimensions. */

    /** Set the correct size and allocate memory. */
    this->m_OptimizationSurface->SetRegions(this->GetSearchSpaceSize());
    this->m_OptimizationSurface->Allocate();
    /** \todo try/catch block around Allocate? */

    /** Set the name of this image on disk. */
    std::string resultImageFormat = "mhd";
    this->m_Configuration->ReadParameter(resultImageFormat, "ResultImageFormat", 0, false);
    makeString.str("");
    makeString << this->GetConfiguration()->GetCommandLineArgument("-out") << "OptimizationSurface."
               << this->GetConfiguration()->GetElastixLevel() << ".R" << level << "." << resultImageFormat;
    this->m_OptimizationSurface->SetOutputFileName(makeString.str().c_str());

    elxout << "Total number of iterations needed in this resolution: " << this->GetNumberOfIterations() << "."
           << std::endl;
  }
  else
  {
    itkExceptionMacro(<< "ERROR: elastix found an error in the search space definition, and is quiting.");
  }

} // end BeforeEachResolution()


/**
 * ***************** AfterEachIteration *************************
 */

template <class TElastix>
void
FullSearch<TElastix>::AfterEachIteration()
{
  /** Print some information. */
  this->GetIterationInfoAt("2:Metric") << this->GetValue();

  this->m_OptimizationSurface->SetPixel(this->GetCurrentIndexInSearchSpace(), this->GetValue());

  SearchSpacePointType currentPoint = this->GetCurrentPointInSearchSpace();
  unsigned int         nrOfSSDims = currentPoint.GetSize();
  NameIteratorType     name_it = this->m_SearchSpaceDimensionNames.begin();

  for (unsigned int dim = 0; dim < nrOfSSDims; ++dim)
  {
    this->GetIterationInfoAt(name_it->second.c_str()) << currentPoint[dim];
    ++name_it;
  }

} // end AfterEachIteration()


/**
 * ***************** AfterEachResolution *************************
 */

template <class TElastix>
void
FullSearch<TElastix>::AfterEachResolution()
{
  // enum StopConditionType {FullRangeSearched,  MetricError };
  std::string stopcondition;

  switch (this->GetStopCondition())
  {
    case FullRangeSearched:
      stopcondition = "The full range has been searched";
      break;

    case MetricError:
      stopcondition = "Error in metric";
      break;

    default:
      stopcondition = "Unknown";
      break;
  }

  /** Print the stopping condition */
  elxout << "Stopping condition: " << stopcondition << "." << std::endl;

  /** Write the optimization surface to disk */
  bool writeSurfaceEachResolution = false;
  this->GetConfiguration()->ReadParameter(
    writeSurfaceEachResolution, "WriteOptimizationSurfaceEachResolution", 0, false);
  if (writeSurfaceEachResolution)
  {
    try
    {
      this->m_OptimizationSurface->Write();
      elxout << "\nThe scanned optimization surface is saved as: " << this->m_OptimizationSurface->GetOutputFileName()
             << std::endl;
    }
    catch (const itk::ExceptionObject & err)
    {
      xl::xout["error"] << "ERROR: Saving " << this->m_OptimizationSurface->GetOutputFileName() << " failed."
                        << std::endl;
      xl::xout["error"] << err << std::endl;
      // do not throw an error, since we would like to go on.
    }
  }

  /** Print the best metric value */
  elxout << '\n' << "Best metric value in this resolution = " << this->GetBestValue() << std::endl;

  /** Print the best index and point */
  SearchSpaceIndexType bestIndex = this->GetBestIndexInSearchSpace();
  SearchSpacePointType bestPoint = this->GetBestPointInSearchSpace();
  unsigned int         nrOfSSDims = bestIndex.GetSize();

  elxout << "Index of the point in the optimization surface image that has the best metric value: [ ";
  for (unsigned int dim = 0; dim < nrOfSSDims; ++dim)
  {
    elxout << bestIndex[dim] << " ";
  }
  elxout << "]" << std::endl;

  elxout << "The corresponding parameter values: [ ";
  for (unsigned int dim = 0; dim < nrOfSSDims; ++dim)
  {
    elxout << bestPoint[dim] << " ";
  }
  elxout << "]\n" << std::endl;

  /** Remove the columns from IterationInfo. */
  NameIteratorType name_it = this->m_SearchSpaceDimensionNames.begin();
  for (unsigned int dim = 0; dim < nrOfSSDims; ++dim)
  {
    this->RemoveTargetCellFromIterationInfo(name_it->second.c_str());
    ++name_it;
  }

  /** Clear the dimension names of the previous resolution's search space. */
  this->m_SearchSpaceDimensionNames.clear();

  /** Clear the full search ranges */
  this->SetSearchSpace(nullptr);

} // end AfterEachResolution()


/**
 * ******************* AfterRegistration ************************
 */
template <class TElastix>
void
FullSearch<TElastix>::AfterRegistration()
{
  /** Print the best metric value. */
  double bestValue = this->GetBestValue();
  elxout << '\n' << "Final metric value  = " << bestValue << std::endl;

} // end AfterRegistration()


/**
 * ************ CheckSearchSpaceRangeDefinition *****************
 */

template <class TElastix>
bool
FullSearch<TElastix>::CheckSearchSpaceRangeDefinition(const std::string & fullFieldName,
                                                      const bool          found,
                                                      const unsigned int  entry_nr) const
{
  /** Complain if not at least one search space dimension has been found,
   * or if a search dimension is not fully specified.
   */
  if (!found && (entry_nr == 0 || (entry_nr % 5 != 0)))
  {
    xl::xout["error"] << "ERROR:\nNo (valid) range specified for the full search optimizer!\n"
                      << "Please define the field (" << fullFieldName
                      << " \"name\" parameter_nr min max stepsize) correctly in the parameter file" << std::endl;
    return false;
  }

  return true;

} // end CheckSearchSpaceRangeDefinition()


} // end namespace elastix

#endif // end #ifndef elxFullSearchOptimizer_hxx
