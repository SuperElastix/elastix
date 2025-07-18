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
#include <itkDeref.h>
#include <iomanip>
#include <sstream>
#include <string>
#include <vnl/vnl_math.h>

namespace elastix
{

/**
 * ********************* Constructor ****************************
 */

template <typename TElastix>
FullSearch<TElastix>::FullSearch()
{
  this->m_OptimizationSurface = nullptr;

} // end Constructor


/**
 * ***************** BeforeRegistration ***********************
 */

template <typename TElastix>
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

template <typename TElastix>
void
FullSearch<TElastix>::BeforeEachResolution()
{
  /** Get the current resolution level.*/
  auto level = static_cast<unsigned int>(this->m_Registration->GetAsITKBaseType()->GetCurrentLevel());

  const Configuration & configuration = itk::Deref(Superclass2::GetConfiguration());

  /** Read FullSearchRange from the parameter file. */

  /** declare variables */
  std::string       name("");
  unsigned int      param_nr = 0;
  double            minimum = 0;
  double            maximum = 0;
  double            stepsize = 0;
  const std::string prefix = "FS";
  unsigned int      entry_nr = 0;
  unsigned int      nrOfSearchSpaceDimensions = 0;
  bool              found = true;
  bool              realGood = true;

  /** Create fullFieldName, which is "FullSearchSpace0" at level 0. */
  const std::string fullFieldName = "FullSearchSpace" + std::to_string(level);

  /** Loop as long as search ranges are defined. */
  while (found)
  {
    /** Try to read (silently) from the parameter file. */
    /** \todo check earlier, in BeforeAll, if the searchspace has been defined. */

    if (realGood && found)
    {
      found = configuration.ReadParameter(name, fullFieldName, entry_nr, false);
      realGood = found || this->CheckSearchSpaceRangeDefinition(fullFieldName, entry_nr);
      ++entry_nr;
    }
    if (realGood && found)
    {
      found = configuration.ReadParameter(param_nr, fullFieldName, entry_nr, false);
      realGood = found || this->CheckSearchSpaceRangeDefinition(fullFieldName, entry_nr);
      ++entry_nr;
    }
    if (realGood && found)
    {
      found = configuration.ReadParameter(minimum, fullFieldName, entry_nr, false);
      realGood = found || this->CheckSearchSpaceRangeDefinition(fullFieldName, entry_nr);
      ++entry_nr;
    }
    if (realGood && found)
    {
      found = configuration.ReadParameter(maximum, fullFieldName, entry_nr, false);
      realGood = found || this->CheckSearchSpaceRangeDefinition(fullFieldName, entry_nr);
      ++entry_nr;
    }
    if (realGood && found)
    {
      found = configuration.ReadParameter(stepsize, fullFieldName, entry_nr, false);
      realGood = found || this->CheckSearchSpaceRangeDefinition(fullFieldName, entry_nr);
      ++entry_nr;
    }

    /** Setup this search range. */
    if (realGood && found)
    {
      /** Setup the Superclass */
      this->AddSearchDimension(param_nr, minimum, maximum, stepsize);

      /** Create name of this dimension. */
      std::ostringstream makeString;
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

    if (const std::string outputDirectoryPath = configuration.GetCommandLineArgument("-out");
        !outputDirectoryPath.empty())
    {
      /** Set the name of this image on disk. */
      std::string resultImageFormat = "mhd";
      configuration.ReadParameter(resultImageFormat, "ResultImageFormat", 0, false);

      std::ostringstream makeString;
      makeString << outputDirectoryPath << "OptimizationSurface." << configuration.GetElastixLevel() << ".R" << level
                 << "." << resultImageFormat;
      this->m_OptimizationSurface->SetOutputFileName(makeString.str().c_str());
    }

    log::info(std::ostringstream{} << "Total number of iterations needed in this resolution: "
                                   << this->GetNumberOfIterations() << ".");
  }
  else
  {
    itkExceptionMacro("ERROR: elastix found an error in the search space definition, and is quiting.");
  }

} // end BeforeEachResolution()


/**
 * ***************** AfterEachIteration *************************
 */

template <typename TElastix>
void
FullSearch<TElastix>::AfterEachIteration()
{
  /** Print some information. */
  this->GetIterationInfoAt("2:Metric") << this->GetValue();

  this->m_OptimizationSurface->SetPixel(this->GetCurrentIndexInSearchSpace(), this->GetValue());

  SearchSpacePointType currentPoint = this->GetCurrentPointInSearchSpace();
  unsigned int         nrOfSSDims = currentPoint.GetSize();
  auto                 name_it = this->m_SearchSpaceDimensionNames.begin();

  for (unsigned int dim = 0; dim < nrOfSSDims; ++dim)
  {
    this->GetIterationInfoAt(name_it->second.c_str()) << currentPoint[dim];
    ++name_it;
  }

} // end AfterEachIteration()


/**
 * ***************** AfterEachResolution *************************
 */

template <typename TElastix>
void
FullSearch<TElastix>::AfterEachResolution()
{
  const Configuration & configuration = itk::Deref(Superclass2::GetConfiguration());

  // enum StopConditionType {FullRangeSearched,  MetricError };
  const std::string stopcondition = [this] {
    switch (this->GetStopCondition())
    {
      case FullRangeSearched:
        return "The full range has been searched";
      case MetricError:
        return "Error in metric";
      default:
        return "Unknown";
    }
  }();

  /** Print the stopping condition */
  log::info(std::ostringstream{} << "Stopping condition: " << stopcondition << ".");

  /** Write the optimization surface to disk */
  bool writeSurfaceEachResolution = false;
  configuration.ReadParameter(writeSurfaceEachResolution, "WriteOptimizationSurfaceEachResolution", 0, false);

  if (writeSurfaceEachResolution && !configuration.GetCommandLineArgument("-out").empty())
  {
    try
    {
      this->m_OptimizationSurface->Write();
      log::info(std::ostringstream{} << "\nThe scanned optimization surface is saved as: "
                                     << this->m_OptimizationSurface->GetOutputFileName());
    }
    catch (const itk::ExceptionObject & err)
    {
      log::error(std::ostringstream{} << "ERROR: Saving " << this->m_OptimizationSurface->GetOutputFileName()
                                      << " failed.\n"
                                      << err);
      // do not throw an error, since we would like to go on.
    }
  }

  /** Print the best metric value */
  log::info(std::ostringstream{} << '\n' << "Best metric value in this resolution = " << this->GetBestValue());

  /** Print the best index and point */
  SearchSpaceIndexType bestIndex = this->GetBestIndexInSearchSpace();
  SearchSpacePointType bestPoint = this->GetBestPointInSearchSpace();
  unsigned int         nrOfSSDims = bestIndex.GetSize();

  std::ostringstream outputStringStream;

  outputStringStream << "Index of the point in the optimization surface image that has the best metric value: [ ";
  for (unsigned int dim = 0; dim < nrOfSSDims; ++dim)
  {
    outputStringStream << bestIndex[dim] << " ";
  }
  outputStringStream << "]\n"

                     << "The corresponding parameter values: [ ";
  for (unsigned int dim = 0; dim < nrOfSSDims; ++dim)
  {
    outputStringStream << bestPoint[dim] << " ";
  }
  outputStringStream << "]\n";
  log::info(outputStringStream.str());

  /** Remove the columns from IterationInfo. */
  auto name_it = this->m_SearchSpaceDimensionNames.begin();
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
template <typename TElastix>
void
FullSearch<TElastix>::AfterRegistration()
{
  /** Print the best metric value. */
  double bestValue = this->GetBestValue();
  log::info(std::ostringstream{} << '\n' << "Final metric value  = " << bestValue);

} // end AfterRegistration()


/**
 * ************ CheckSearchSpaceRangeDefinition *****************
 */

template <typename TElastix>
bool
FullSearch<TElastix>::CheckSearchSpaceRangeDefinition(const std::string & fullFieldName,
                                                      const unsigned int  entry_nr) const
{
  /** Complain if not at least one search space dimension has been found,
   * or if a search dimension is not fully specified.
   */
  if (entry_nr == 0 || (entry_nr % 5 != 0))
  {
    log::error(std::ostringstream{} << "ERROR:\nNo (valid) range specified for the full search optimizer!\n"
                                    << "Please define the field (" << fullFieldName
                                    << " \"name\" parameter_nr min max stepsize) correctly in the parameter file");
    return false;
  }

  return true;

} // end CheckSearchSpaceRangeDefinition()


} // end namespace elastix

#endif // end #ifndef elxFullSearchOptimizer_hxx
