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
#ifndef elxMultiBSplineTransformWithNormal_hxx
#define elxMultiBSplineTransformWithNormal_hxx

#include "elxMultiBSplineTransformWithNormal.h"

#include "itkImageFileReader.h"
#include "itkImageRegionExclusionConstIteratorWithIndex.h"
#include "itkResampleImageFilter.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itksys/System.h"
#include <vnl/vnl_math.h>

namespace elastix
{

/**
 * ************ InitializeBSplineTransform ***************
 */

template <class TElastix>
unsigned int
MultiBSplineTransformWithNormal<TElastix>::InitializeBSplineTransform()
{
  /** Initialize the right BSplineTransform and GridScheduleComputer. */
  this->m_GridScheduleComputer = GridScheduleComputerType::New();
  this->m_GridScheduleComputer->SetBSplineOrder(this->m_SplineOrder);

  /*
  if ( this->m_SplineOrder == 1)
  {
    this->m_MultiBSplineTransformWithNormal = MultiBSplineTransformWithNormalLinearType::New();
  }
  else if ( this->m_SplineOrder == 2)
  {
    this->m_MultiBSplineTransformWithNormal = MultiBSplineTransformWithNormalQuadraticType::New();
  }
  else */
  if (this->m_SplineOrder == 3)
  {
    this->m_MultiBSplineTransformWithNormal = MultiBSplineTransformWithNormalCubicType::New();
  }
  else
  {
    itkExceptionMacro(<< "ERROR: The provided spline order is not supported.");
    return 1;
  }

  this->SetCurrentTransform(this->m_MultiBSplineTransformWithNormal);
  this->m_GridUpsampler = GridUpsamplerType::New();
  this->m_GridUpsampler->SetBSplineOrder(m_SplineOrder);

  return 0;
} // end InitializeBSplineTransform()


/**
 * ******************* BeforeAll ***********************
 */

template <class TElastix>
int
MultiBSplineTransformWithNormal<TElastix>::BeforeAll()
{
  /** Read spline order and periodicity setting from configuration file. */
  this->m_SplineOrder = 3;
  this->GetConfiguration()->ReadParameter(
    this->m_SplineOrder, "BSplineTransformSplineOrder", this->GetComponentLabel(), 0, 0, true);

  this->m_LabelsPath = this->m_Configuration->GetCommandLineArgument("-labels");
  if (!this->m_LabelsPath.empty())
  {
    m_Labels = itk::ReadImage<ImageLabelType>(m_LabelsPath);
  }
  else
  {
    xl::xout["error"] << "ERROR: The MultiBSplineTransformWithNormal need a -labels command line option"
                      << " that indicates where to find the sliding objects segmentation." << std::endl;
    itkExceptionMacro(<< "ERROR: Missing -labels argument!");
  }

  return InitializeBSplineTransform();

} // end BeforeAll()


/**
 * ******************* BeforeRegistration ***********************
 */

template <class TElastix>
void
MultiBSplineTransformWithNormal<TElastix>::BeforeRegistration()
{
  /** Set initial transform parameters to a 1x1x1 grid, with deformation (0,0,0).
   * In the method BeforeEachResolution() this will be replaced by the right grid size.
   * This seems not logical, but it is required, since the registration
   * class checks if the number of parameters in the transform is equal to
   * the number of parameters in the registration class. This check is done
   * before calling the BeforeEachResolution() methods.
   */

  /** Task 1 - Set the Grid. */

  /** Set gridsize for large dimension to 4 to prevent errors when checking
   * on support region size.
   */
  SizeType gridsize = SizeType::Filled(1);
  gridsize.SetElement(gridsize.GetSizeDimension() - 1, 4);

  /** Set it all. */
  this->m_MultiBSplineTransformWithNormal->SetGridRegion(RegionType(gridsize));
  this->m_MultiBSplineTransformWithNormal->SetGridSpacing(SpacingType(1.0));
  this->m_MultiBSplineTransformWithNormal->SetGridOrigin(OriginType());

  /** Task 2 - Give the registration an initial parameter-array. */
  this->m_Registration->GetAsITKBaseType()->SetInitialTransformParameters(
    ParametersType(this->GetNumberOfParameters(), 0.0));

  /** Precompute the B-spline grid regions. */
  this->PreComputeGridInformation();

} // end BeforeRegistration()


/**
 * ***************** BeforeEachResolution ***********************
 */

template <class TElastix>
void
MultiBSplineTransformWithNormal<TElastix>::BeforeEachResolution()
{
  /** What is the current resolution level? */
  unsigned int level = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

  /** Define the grid. */
  if (level == 0)
  {
    this->InitializeTransform();
  }
  else
  {
    /** Upsample the B-spline grid, if required. */
    this->IncreaseScale();
  }

} // end BeforeEachResolution()


/**
 * ******************** PreComputeGridInformation ***********************
 */

template <class TElastix>
void
MultiBSplineTransformWithNormal<TElastix>::PreComputeGridInformation()
{
  /** Get the total number of resolution levels. */
  unsigned int nrOfResolutions = this->m_Registration->GetAsITKBaseType()->GetNumberOfLevels();

  /** Set up grid schedule computer with image info. */
  if (this->GetElastix()->GetFixedMask())
  {
    this->m_GridScheduleComputer->SetImageOrigin(this->GetElastix()->GetFixedMask()->GetOrigin());
    this->m_GridScheduleComputer->SetImageSpacing(this->GetElastix()->GetFixedMask()->GetSpacing());
    this->m_GridScheduleComputer->SetImageDirection(this->GetElastix()->GetFixedMask()->GetDirection());
    this->m_GridScheduleComputer->SetImageRegion(this->GetElastix()->GetFixedMask()->GetLargestPossibleRegion());
  }
  else
  {
    this->m_GridScheduleComputer->SetImageOrigin(this->GetElastix()->GetFixedImage()->GetOrigin());
    this->m_GridScheduleComputer->SetImageSpacing(this->GetElastix()->GetFixedImage()->GetSpacing());
    this->m_GridScheduleComputer->SetImageDirection(this->GetElastix()->GetFixedImage()->GetDirection());
    this->m_GridScheduleComputer->SetImageRegion(this->GetElastix()->GetFixedImage()->GetLargestPossibleRegion());
  }

  /** Take the initial transform only into account, if composition is used. */
  if (this->GetUseComposition())
  {
    this->m_GridScheduleComputer->SetInitialTransform(this->Superclass1::GetInitialTransform());
  }

  /** Get the grid spacing schedule from the parameter file.
   *
   * Method 1: The user specifies "FinalGridSpacingInVoxels"
   * Method 2: The user specifies "FinalGridSpacingInPhysicalUnits"
   *
   * Method 1 and 2 additionally take the "GridSpacingSchedule".
   * The GridSpacingSchedule is defined by downsampling factors
   * for each resolution, for each dimension (just like the image
   * pyramid schedules). So, for 2D images, and 3 resulutions,
   * we can specify:
   * (GridSpacingSchedule 4.0 4.0 2.0 2.0 1.0 1.0)
   * Which is the default schedule, if no GridSpacingSchedule is supplied.
   */

  /** Determine which method is used. */
  bool         method1 = false;
  unsigned int count1 = this->m_Configuration->CountNumberOfParameterEntries("FinalGridSpacingInVoxels");
  if (count1 > 0)
  {
    method1 = true;
  }

  bool         method2 = false;
  unsigned int count2 = this->m_Configuration->CountNumberOfParameterEntries("FinalGridSpacingInPhysicalUnits");
  if (count2 > 0)
  {
    method2 = true;
  }

  /** Throw an exception if both methods are used. */
  if (count1 > 0 && count2 > 0)
  {
    itkExceptionMacro(<< "ERROR: You can not specify both \"FinalGridSpacingInVoxels\""
                         " and \"FinalGridSpacingInPhysicalUnits\" in the parameter file.");
  }

  /** Declare variables and set defaults. */
  SpacingType finalGridSpacingInVoxels;
  SpacingType finalGridSpacingInPhysicalUnits;
  finalGridSpacingInVoxels.Fill(16.0);
  finalGridSpacingInPhysicalUnits.Fill(8.0);

  /** Method 1: Read the FinalGridSpacingInVoxels. */
  if (method1)
  {
    for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
    {
      this->m_Configuration->ReadParameter(
        finalGridSpacingInVoxels[dim], "FinalGridSpacingInVoxels", this->GetComponentLabel(), dim, 0);
    }

    /** Compute the grid spacing in physical units. */
    for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
    {
      finalGridSpacingInPhysicalUnits[dim] =
        finalGridSpacingInVoxels[dim] * this->GetElastix()->GetFixedImage()->GetSpacing()[dim];
    }
  }

  /** Method 2: Read the FinalGridSpacingInPhysicalUnits. */
  if (method2)
  {
    for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
    {
      this->m_Configuration->ReadParameter(
        finalGridSpacingInPhysicalUnits[dim], "FinalGridSpacingInPhysicalUnits", this->GetComponentLabel(), dim, 0);
    }
  }

  /** Set up a default grid spacing schedule. */
  this->m_GridScheduleComputer->SetDefaultSchedule(nrOfResolutions, 2.0);
  GridScheduleType gridSchedule;
  this->m_GridScheduleComputer->GetSchedule(gridSchedule);

  /** Read what the user has specified. This overrules everything. */
  count2 = this->m_Configuration->CountNumberOfParameterEntries("GridSpacingSchedule");
  unsigned int entry_nr = 0;
  if (count2 == 0)
  {
    // keep the default schedule
  }
  else if (count2 == nrOfResolutions)
  {
    for (unsigned int res = 0; res < nrOfResolutions; ++res)
    {
      for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
      {
        this->m_Configuration->ReadParameter(gridSchedule[res][dim], "GridSpacingSchedule", entry_nr, false);
      }
      ++entry_nr;
    }
  }
  else if (count2 == nrOfResolutions * SpaceDimension)
  {
    for (unsigned int res = 0; res < nrOfResolutions; ++res)
    {
      for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
      {
        this->m_Configuration->ReadParameter(gridSchedule[res][dim], "GridSpacingSchedule", entry_nr, false);
        ++entry_nr;
      }
    }
  }
  else
  {
    xl::xout["error"] << "ERROR: Invalid GridSpacingSchedule! The number of entries behind the GridSpacingSchedule "
                         "option should equal the numberOfResolutions, or the numberOfResolutions * ImageDimension."
                      << std::endl;
    itkExceptionMacro(<< "ERROR: Invalid GridSpacingSchedule!");
  }

  /** Set the grid schedule and final grid spacing in the schedule computer. */
  this->m_GridScheduleComputer->SetFinalGridSpacing(finalGridSpacingInPhysicalUnits);
  this->m_GridScheduleComputer->SetSchedule(gridSchedule);

  /** Compute the necessary information. */
  this->m_GridScheduleComputer->ComputeBSplineGrid();

} // end PreComputeGridInformation()


/**
 * ******************** InitializeTransform ***********************
 *
 * Set the size of the initial control point grid and initialize
 * the parameters to 0.
 */

template <class TElastix>
void
MultiBSplineTransformWithNormal<TElastix>::InitializeTransform()
{
  /** Compute the B-spline grid region, origin, and spacing. */
  RegionType    gridRegion;
  OriginType    gridOrigin;
  SpacingType   gridSpacing;
  DirectionType gridDirection;
  this->m_GridScheduleComputer->GetBSplineGrid(0, gridRegion, gridSpacing, gridOrigin, gridDirection);

  /** Set it in the BSplineTransform. */
  this->m_MultiBSplineTransformWithNormal->SetGridRegion(gridRegion);
  this->m_MultiBSplineTransformWithNormal->SetGridSpacing(gridSpacing);
  this->m_MultiBSplineTransformWithNormal->SetGridOrigin(gridOrigin);
  this->m_MultiBSplineTransformWithNormal->SetGridDirection(gridDirection);
  this->m_MultiBSplineTransformWithNormal->SetLabels(m_Labels);
  this->m_MultiBSplineTransformWithNormal->UpdateLocalBases();

  /** Set initial parameters for the first resolution to 0.0. */
  ParametersType initialParameters(this->GetNumberOfParameters());
  initialParameters.Fill(0.0);
  this->m_Registration->GetAsITKBaseType()->SetInitialTransformParametersOfNextLevel(initialParameters);

} // end InitializeTransform()


/**
 * *********************** IncreaseScale ************************
 *
 * Upsample the grid of control points.
 */

template <class TElastix>
void
MultiBSplineTransformWithNormal<TElastix>::IncreaseScale()
{
  /** What is the current resolution level? */
  unsigned int level = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

  /** The current grid. */
  OriginType    currentGridOrigin = this->m_MultiBSplineTransformWithNormal->GetGridOrigin();
  SpacingType   currentGridSpacing = this->m_MultiBSplineTransformWithNormal->GetGridSpacing();
  RegionType    currentGridRegion = this->m_MultiBSplineTransformWithNormal->GetGridRegion();
  DirectionType currentGridDirection = this->m_MultiBSplineTransformWithNormal->GetGridDirection();

  /** The new required grid. */
  OriginType    requiredGridOrigin;
  SpacingType   requiredGridSpacing;
  RegionType    requiredGridRegion;
  DirectionType requiredGridDirection;
  this->m_GridScheduleComputer->GetBSplineGrid(
    level, requiredGridRegion, requiredGridSpacing, requiredGridOrigin, requiredGridDirection);

  /** Get the latest transform parameters. */
  ParametersType latestParameters = this->m_Registration->GetAsITKBaseType()->GetLastTransformParameters();

  /** Setup the GridUpsampler. */
  /* FIXME : Need MultiGrid Upsampler or loop on grids */
  this->m_GridUpsampler->SetCurrentGridOrigin(currentGridOrigin);
  this->m_GridUpsampler->SetCurrentGridSpacing(currentGridSpacing);
  this->m_GridUpsampler->SetCurrentGridRegion(currentGridRegion);
  this->m_GridUpsampler->SetCurrentGridDirection(currentGridDirection);
  this->m_GridUpsampler->SetRequiredGridOrigin(requiredGridOrigin);
  this->m_GridUpsampler->SetRequiredGridSpacing(requiredGridSpacing);
  this->m_GridUpsampler->SetRequiredGridRegion(requiredGridRegion);
  this->m_GridUpsampler->SetRequiredGridDirection(requiredGridDirection);

  using VectorType = itk::Vector<double, Self::SpaceDimension>;

  using BaseType = itk::Vector<VectorType, Self::SpaceDimension>;
  using ImageBaseType = itk::Image<BaseType, Self::SpaceDimension>;
  using ImageBasePointer = typename ImageBaseType::Pointer;
  using BaseContainer = typename ImageBaseType::PixelContainer;

  ImageBasePointer      Bases = this->m_MultiBSplineTransformWithNormal->GetLocalBases();
  const BaseContainer & bases = *Bases->GetPixelContainer();

  unsigned ParametersPerDimension = this->m_MultiBSplineTransformWithNormal->GetNumberOfParametersPerDimension();

  ParametersType upsampledNormalParameters, tmpParameters;
  tmpParameters.SetSize(ParametersPerDimension * SpaceDimension);
  for (unsigned int i = 0; i < ParametersPerDimension; ++i)
  {
    VectorType tmp = latestParameters.GetElement(i) * bases[i][0];
    for (unsigned int d = 0; d < SpaceDimension; ++d)
    {
      tmpParameters.SetElement(i + d * ParametersPerDimension, tmp[d]);
    }
  }
  this->m_GridUpsampler->UpsampleParameters(tmpParameters, upsampledNormalParameters);

  this->m_MultiBSplineTransformWithNormal->SetGridOrigin(requiredGridOrigin);
  this->m_MultiBSplineTransformWithNormal->SetGridSpacing(requiredGridSpacing);
  this->m_MultiBSplineTransformWithNormal->SetGridRegion(requiredGridRegion);
  this->m_MultiBSplineTransformWithNormal->SetGridDirection(requiredGridDirection);
  this->m_MultiBSplineTransformWithNormal->UpdateLocalBases();

  ImageBasePointer      new_Bases = this->m_MultiBSplineTransformWithNormal->GetLocalBases();
  const BaseContainer & new_bases = *new_Bases->GetPixelContainer();

  using ImageLabelType = itk::Image<unsigned char, Self::SpaceDimension>;
  typename ImageLabelType::Pointer labels1 = this->m_MultiBSplineTransformWithNormal->GetLabels();
  typename itk::ResampleImageFilter<ImageLabelType, ImageLabelType>::Pointer filter =
    itk::ResampleImageFilter<ImageLabelType, ImageLabelType>::New();
  filter->SetInterpolator(itk::NearestNeighborInterpolateImageFunction<ImageLabelType, double>::New());
  filter->SetInput(labels1);
  filter->SetOutputParametersFromImage(new_Bases);
  filter->Update();

  using LabelContainer = typename ImageLabelType::PixelContainer;
  const LabelContainer & labels = *filter->GetOutput()->GetPixelContainer();

  unsigned new_ParametersPerDimension = this->m_MultiBSplineTransformWithNormal->GetNumberOfParametersPerDimension();

  unsigned char  NbLabels = this->m_MultiBSplineTransformWithNormal->GetNbLabels();
  ParametersType upsampledParameters;
  upsampledParameters.SetSize((1 + (SpaceDimension - 1) * NbLabels) * new_ParametersPerDimension);
  upsampledParameters.Fill(0.0);

  for (unsigned int l = 1; l <= NbLabels; ++l)
  {
    ParametersType upsampledLabelParameters;
    for (unsigned int i = 0; i < ParametersPerDimension; ++i)
    {
      VectorType tmp;
      for (unsigned int d = 0; d < SpaceDimension; ++d)
      {
        tmp[d] = 0;
      }

      for (unsigned int d = 1; d < SpaceDimension; ++d)
      {
        tmp +=
          bases[i][d] * latestParameters.GetElement(i + ((SpaceDimension - 1) * (l - 1) + d) * ParametersPerDimension);
      }

      for (unsigned int d = 0; d < SpaceDimension; ++d)
      {
        tmpParameters.SetElement(i + d * ParametersPerDimension, tmp[d]);
      }
    }
    this->m_GridUpsampler->UpsampleParameters(tmpParameters, upsampledLabelParameters);

    // Redecompose N / U / V
    for (unsigned int i = 0; i < new_ParametersPerDimension; ++i)
    {
      VectorType tmp;
      for (unsigned int d = 0; d < SpaceDimension; ++d)
      {
        tmp[d] = upsampledLabelParameters.GetElement(i + d * new_ParametersPerDimension);
      }

      if (static_cast<unsigned int>(labels[i] + 1) == l)
      {
        for (unsigned int d = 0; d < SpaceDimension; ++d)
        {
          tmp[d] += upsampledNormalParameters.GetElement(i + d * new_ParametersPerDimension);
        }
      }

      itk::Matrix<double, SpaceDimension, SpaceDimension> P;
      for (unsigned int a = 0; a < SpaceDimension; ++a)
      {
        for (unsigned int b = 0; b < SpaceDimension; ++b)
        {
          P[a][b] = new_bases[i][a][b];
        }
      }
      tmp = P * tmp;
      if (static_cast<unsigned int>(labels[i] + 1) == l)
      {
        upsampledParameters.SetElement(i, tmp[0]);
      }
      for (unsigned int k = 0; k < (SpaceDimension - 1); ++k)
      {
        upsampledParameters.SetElement(i + ((SpaceDimension - 1) * (l - 1) + 1 + k) * new_ParametersPerDimension,
                                       tmp[k + 1]);
      }
    }
  }

  /** Set the initial parameters for the next level. */
  this->m_Registration->GetAsITKBaseType()->SetInitialTransformParametersOfNextLevel(upsampledParameters);

  /** Set the parameters in the BsplineTransform. */
  this->m_MultiBSplineTransformWithNormal->SetParameters(
    this->m_Registration->GetAsITKBaseType()->GetInitialTransformParametersOfNextLevel());

} // end IncreaseScale()


/**
 * ************************* ReadFromFile ************************
 */

template <class TElastix>
void
MultiBSplineTransformWithNormal<TElastix>::ReadFromFile()
{
  /** Read spline order and periodicity settings and initialize BSplineTransform. */
  this->m_SplineOrder = 3;
  this->GetConfiguration()->ReadParameter(
    this->m_SplineOrder, "BSplineTransformSplineOrder", this->GetComponentLabel(), 0, 0);
  this->InitializeBSplineTransform();

  /** Read and Set the Grid: this is a BSplineTransform specific task. */

  /** Declarations. Everything filled with default values.*/
  SizeType      gridsize = SizeType::Filled(1);
  IndexType     gridindex = { { 0 } };
  SpacingType   gridspacing(1.0);
  OriginType    gridorigin{};
  DirectionType griddirection = DirectionType::GetIdentity();

  /** Get GridSize, GridIndex, GridSpacing and GridOrigin. */
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    this->m_Configuration->ReadParameter(gridsize[i], "GridSize", i);
    this->m_Configuration->ReadParameter(gridindex[i], "GridIndex", i);
    this->m_Configuration->ReadParameter(gridspacing[i], "GridSpacing", i);
    this->m_Configuration->ReadParameter(gridorigin[i], "GridOrigin", i);
    for (unsigned int j = 0; j < SpaceDimension; ++j)
    {
      this->m_Configuration->ReadParameter(griddirection(j, i), "GridDirection", i * SpaceDimension + j);
    }
  }

  /** Set it all. */
  this->m_MultiBSplineTransformWithNormal->SetGridRegion(RegionType(gridindex, gridsize));
  this->m_MultiBSplineTransformWithNormal->SetGridSpacing(gridspacing);
  this->m_MultiBSplineTransformWithNormal->SetGridOrigin(gridorigin);
  this->m_MultiBSplineTransformWithNormal->SetGridDirection(griddirection);

  /** Read the labels map */
  this->GetConfiguration()->ReadParameter(
    this->m_LabelsPath, "MultiBSplineTransformWithNormalLabels", this->GetComponentLabel(), 0, 0);
  if (!this->m_LabelsPath.empty())
  {
    this->m_Labels = itk::ReadImage<ImageLabelType>(m_LabelsPath);
  }
  this->m_MultiBSplineTransformWithNormal->SetLabels(this->m_Labels);
  this->m_MultiBSplineTransformWithNormal->UpdateLocalBases();

  /** Call the ReadFromFile from the TransformBase.
   * This must be done after setting the Grid, because later the
   * ReadFromFile from TransformBase calls SetParameters, which
   * checks the parameter-size, which is based on the GridSize.
   */
  this->Superclass2::ReadFromFile();

} // end ReadFromFile()


/**
 * ************************* CustomizeTransformParametersMap ************************
 */

template <class TElastix>
auto
MultiBSplineTransformWithNormal<TElastix>::CreateDerivedTransformParametersMap() const -> ParameterMapType
{
  const auto & itkTransform = *m_MultiBSplineTransformWithNormal;
  const auto   gridRegion = itkTransform.GetGridRegion();

  return { { "GridSize", Conversion::ToVectorOfStrings(gridRegion.GetSize()) },
           { "GridIndex", Conversion::ToVectorOfStrings(gridRegion.GetIndex()) },
           { "GridSpacing", Conversion::ToVectorOfStrings(itkTransform.GetGridSpacing()) },
           { "GridOrigin", Conversion::ToVectorOfStrings(itkTransform.GetGridOrigin()) },
           { "GridDirection", Conversion::ToVectorOfStrings(itkTransform.GetGridDirection()) },
           { "BSplineTransformSplineOrder", { Conversion::ToString(m_SplineOrder) } },
           { "MultiBSplineTransformWithNormalLabels", { itksys::SystemTools::CollapseFullPath(m_LabelsPath) } } };

} // end CustomizeTransformParametersMap()


/**
 * *********************** SetOptimizerScales ***********************
 *
 * Set the optimizer scales of the edge coefficients to infinity.
 */

template <class TElastix>
void
MultiBSplineTransformWithNormal<TElastix>::SetOptimizerScales(const unsigned int edgeWidth)
{
  /** Some typedefs. */
  using IteratorType = itk::ImageRegionExclusionConstIteratorWithIndex<ImageType>;
  using ITKRegistrationType = typename RegistrationType::ITKBaseType;
  using OptimizerType = typename ITKRegistrationType::OptimizerType;
  using ScalesType = typename OptimizerType::ScalesType;
  using ScalesValueType = typename ScalesType::ValueType;

  /** Define new scales. */
  const unsigned long   numberOfParameters = this->m_MultiBSplineTransformWithNormal->GetNumberOfParameters();
  const unsigned long   offset = numberOfParameters / SpaceDimension;
  ScalesType            newScales(numberOfParameters, ScalesValueType{ 1.0 });
  const ScalesValueType infScale = 10000.0;

  if (edgeWidth == 0)
  {
    /** Just set the unit scales into the optimizer. */
    this->m_Registration->GetAsITKBaseType()->GetModifiableOptimizer()->SetScales(newScales);
    return;
  }

  /** Get the grid region information and create a fake coefficient image. */
  RegionType   gridregion = this->m_MultiBSplineTransformWithNormal->GetGridRegion();
  SizeType     gridsize = gridregion.GetSize();
  IndexType    gridindex = gridregion.GetIndex();
  ImagePointer coeff = ImageType::New();
  coeff->SetRegions(gridregion);
  coeff->Allocate();

  /** Determine inset region. (so, the region with active parameters). */
  RegionType insetgridregion;
  SizeType   insetgridsize;
  IndexType  insetgridindex;
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    insetgridsize[i] = static_cast<unsigned int>(std::max(0, static_cast<int>(gridsize[i] - 2 * edgeWidth)));
    if (insetgridsize[i] == 0)
    {
      xl::xout["error"] << "ERROR: you specified a PassiveEdgeWidth of " << edgeWidth
                        << ", while the total grid size in dimension " << i << " is only " << gridsize[i] << "."
                        << std::endl;
      itkExceptionMacro(<< "ERROR: the PassiveEdgeWidth is too large!");
    }
    insetgridindex[i] = gridindex[i] + edgeWidth;
  }
  insetgridregion.SetSize(insetgridsize);
  insetgridregion.SetIndex(insetgridindex);

  /** Set up iterator over the coefficient image. */
  IteratorType cIt(coeff, coeff->GetLargestPossibleRegion());
  cIt.SetExclusionRegion(insetgridregion);
  cIt.GoToBegin();

  /** Set the scales to infinity that correspond to edge coefficients
   * This (hopefully) makes sure they are not optimized during registration.
   */
  while (!cIt.IsAtEnd())
  {
    const IndexType &   index = cIt.GetIndex();
    const unsigned long baseOffset = coeff->ComputeOffset(index);
    for (unsigned int i = 0; i < SpaceDimension; ++i)
    {
      const unsigned int scalesIndex = static_cast<unsigned int>(baseOffset + i * offset);
      newScales[scalesIndex] = infScale;
    }
    ++cIt;
  }

  /** Set the scales into the optimizer. */
  this->m_Registration->GetAsITKBaseType()->GetModifiableOptimizer()->SetScales(newScales);

} // end SetOptimizerScales()


} // end namespace elastix

#endif // end #ifndef elxMultiBSplineTransformWithNormal_hxx
