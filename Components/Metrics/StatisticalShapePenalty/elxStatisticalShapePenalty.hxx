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
#ifndef elxStatisticalShapePenalty_hxx
#define elxStatisticalShapePenalty_hxx

#include "elxStatisticalShapePenalty.h"
#include "itkTransformixInputPointFileReader.h"

#include "itkPointSet.h"
#include "itkDefaultStaticMeshTraits.h"
#include "itkTransformMeshFilter.h"
#include <itkMesh.h>
#include <itkMeshFileReader.h>

#include <fstream>
#include <typeinfo>

namespace elastix
{
/**
 * ******************* Initialize ***********************
 */

template <class TElastix>
void
StatisticalShapePenalty<TElastix>::Initialize()
{
  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();
  timer.Stop();
  elxout << "Initialization of StatisticalShape metric took: " << static_cast<long>(timer.GetMean() * 1000) << " ms."
         << std::endl;

} // end Initialize()


/**
 * ***************** BeforeRegistration ***********************
 */

template <class TElastix>
void
StatisticalShapePenalty<TElastix>::BeforeRegistration()
{
  /** Get and set NormalizedShapeModel. Default TRUE. */
  bool normalizedShapeModel = true;
  this->GetConfiguration()->ReadParameter(normalizedShapeModel, "NormalizedShapeModel", 0, 0);
  this->SetNormalizedShapeModel(normalizedShapeModel);

  /** Get and set NormalizedShapeModel. Default TRUE. */
  int shapeModelCalculation = 0;
  this->GetConfiguration()->ReadParameter(shapeModelCalculation, "ShapeModelCalculation", 0, 0);
  this->SetShapeModelCalculation(shapeModelCalculation);

  /** Read and set the fixed pointset. */
  std::string                            fixedName = this->GetConfiguration()->GetCommandLineArgument("-fp");
  typename PointSetType::Pointer         fixedPointSet; // default-constructed (null)
  const typename ImageType::ConstPointer fixedImage = this->GetElastix()->GetFixedImage();
  const unsigned int                     nrOfFixedPoints = this->ReadShape(fixedName, fixedPointSet, fixedImage);
  this->SetFixedPointSet(fixedPointSet);

  // itkCombinationImageToImageMetric.hxx checks if metric base class is ImageMetricType or PointSetMetricType.
  // This class is derived from SingleValuedPointSetToPointSetMetric which needs a moving pointset.
  this->SetMovingPointSet(fixedPointSet);
  // TODO: make itkCombinationImageToImageMetric check for a base class metric that doesn't use an image or moving
  // pointset.

  /** Read meanVector filename. */
  std::string                meanVectorName = this->GetConfiguration()->GetCommandLineArgument("-mean");
  std::ifstream              datafile;
  vnl_vector<double> * const meanVector = new vnl_vector<double>();
  datafile.open(meanVectorName.c_str());
  if (datafile.is_open())
  {
    meanVector->read_ascii(datafile);
    datafile.close();
    datafile.clear();
    elxout << " meanVector " << meanVectorName << " read" << std::endl;
  }
  else
  {
    itkExceptionMacro(<< "Unable to open meanVector file: " << meanVectorName);
  }
  this->SetMeanVector(meanVector);

  /** Check. */
  if (normalizedShapeModel)
  {
    if (nrOfFixedPoints * Self::FixedPointSetDimension != meanVector->size() - Self::FixedPointSetDimension - 1)
    {
      itkExceptionMacro(<< "ERROR: the number of elements in the meanVector (" << meanVector->size()
                        << ") does not match the number of points of the fixed pointset (" << nrOfFixedPoints
                        << ") times the point dimensionality (" << Self::FixedPointSetDimension
                        << ") plus a Centroid of dimension " << Self::FixedPointSetDimension << " plus a size element");
    }
  }
  else
  {
    if (nrOfFixedPoints * Self::FixedPointSetDimension != meanVector->size())
    {
      itkExceptionMacro(<< "ERROR: the number of elements in the meanVector (" << meanVector->size()
                        << ") does not match the number of points of the fixed pointset (" << nrOfFixedPoints
                        << ") times the point dimensionality (" << Self::FixedPointSetDimension << ")");
    }
  }

  /** Read covariance matrix filename. */
  std::string covarianceMatrixName = this->GetConfiguration()->GetCommandLineArgument("-covariance");

  vnl_matrix<double> * const covarianceMatrix = new vnl_matrix<double>();

  datafile.open(covarianceMatrixName.c_str());
  if (datafile.is_open())
  {
    covarianceMatrix->read_ascii(datafile);
    datafile.close();
    datafile.clear();
    elxout << "covarianceMatrix " << covarianceMatrixName << " read" << std::endl;
  }
  else
  {
    itkExceptionMacro(<< "Unable to open covarianceMatrix file: " << covarianceMatrixName);
  }
  this->SetCovarianceMatrix(covarianceMatrix);

  /** Read eigenvector matrix filename. */
  std::string eigenVectorsName = this->GetConfiguration()->GetCommandLineArgument("-evectors");

  vnl_matrix<double> * const eigenVectors = new vnl_matrix<double>();

  datafile.open(eigenVectorsName.c_str());
  if (datafile.is_open())
  {
    eigenVectors->read_ascii(datafile);
    datafile.close();
    datafile.clear();
    elxout << "eigenvectormatrix " << eigenVectorsName << " read" << std::endl;
  }
  else
  {
    // \todo: remove outcommented code:
    // itkExceptionMacro( << "Unable to open EigenVectors file: " << eigenVectorsName);
  }
  this->SetEigenVectors(eigenVectors);

  /** Read eigenvalue vector filename. */
  std::string                eigenValuesName = this->GetConfiguration()->GetCommandLineArgument("-evalues");
  vnl_vector<double> * const eigenValues = new vnl_vector<double>();
  datafile.open(eigenValuesName.c_str());
  if (datafile.is_open())
  {
    eigenValues->read_ascii(datafile);
    datafile.close();
    datafile.clear();
    elxout << "eigenvaluevector " << eigenValuesName << " read" << std::endl;
  }
  else
  {
    // itkExceptionMacro( << "Unable to open EigenValues file: " << eigenValuesName);
  }
  this->SetEigenValues(eigenValues);

} // end BeforeRegistration()


/**
 * ***************** BeforeEachResolution ***********************
 */

template <class TElastix>
void
StatisticalShapePenalty<TElastix>::BeforeEachResolution()
{
  /** Get the current resolution level. */
  unsigned int level = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

  /** Get and set ShrinkageIntensity. Default 0.5. */
  double shrinkageIntensity = 0.5;
  this->GetConfiguration()->ReadParameter(
    shrinkageIntensity, "ShrinkageIntensity", this->GetComponentLabel(), level, 0);

  if (this->GetShrinkageIntensity() != shrinkageIntensity)
  {
    this->SetShrinkageIntensityNeedsUpdate(true);
  }
  this->SetShrinkageIntensity(shrinkageIntensity);

  /** Get and set BaseVariance. Default 1000. */
  double baseVariance = 1000;
  this->GetConfiguration()->ReadParameter(baseVariance, "BaseVariance", this->GetComponentLabel(), level, 0);

  if (this->GetBaseVariance() != baseVariance)
  {
    this->SetBaseVarianceNeedsUpdate(true);
  }
  this->SetBaseVariance(baseVariance);

  /** Get and set CentroidXVariance. Default 10. */
  double centroidXVariance = 10;
  this->GetConfiguration()->ReadParameter(centroidXVariance, "CentroidXVariance", this->GetComponentLabel(), level, 0);

  if (this->GetCentroidXVariance() != centroidXVariance)
  {
    this->SetVariancesNeedsUpdate(true);
  }
  this->SetCentroidXVariance(centroidXVariance);

  /** Get and set CentroidYVariance. Default 10. */
  double centroidYVariance = 10;
  this->GetConfiguration()->ReadParameter(centroidYVariance, "CentroidYVariance", this->GetComponentLabel(), level, 0);

  if (this->GetCentroidYVariance() != centroidYVariance)
  {
    this->SetVariancesNeedsUpdate(true);
  }
  this->SetCentroidYVariance(centroidYVariance);

  /** Get and set CentroidZVariance. Default 10. */
  double centroidZVariance = 10;
  this->GetConfiguration()->ReadParameter(centroidZVariance, "CentroidZVariance", this->GetComponentLabel(), level, 0);

  if (this->GetCentroidZVariance() != centroidZVariance)
  {
    this->SetVariancesNeedsUpdate(true);
  }
  this->SetCentroidZVariance(centroidZVariance);

  /** Get and set SizeVariance. Default 10. */
  double sizeVariance = 10;
  this->GetConfiguration()->ReadParameter(sizeVariance, "SizeVariance", this->GetComponentLabel(), level, 0);

  if (this->GetSizeVariance() != sizeVariance)
  {
    this->SetVariancesNeedsUpdate(true);
  }
  this->SetSizeVariance(sizeVariance);

  /** Get and set CutOffValue. Default 0. */
  double cutOffValue = 0;
  this->GetConfiguration()->ReadParameter(cutOffValue, "CutOffValue", this->GetComponentLabel(), level, 0);
  this->SetCutOffValue(cutOffValue);

  /** Get and set CutOffSharpness. Default 2. */
  double cutOffSharpness = 2.0;
  this->GetConfiguration()->ReadParameter(cutOffSharpness, "CutOffSharpness", this->GetComponentLabel(), level, 0);
  this->SetCutOffSharpness(cutOffSharpness);

} // end BeforeEachResolution()


/**
 * ***************** ReadLandmarks ***********************
 */

template <class TElastix>
unsigned int
StatisticalShapePenalty<TElastix>::ReadLandmarks(const std::string &                    landmarkFileName,
                                                 typename PointSetType::Pointer &       pointSet,
                                                 const typename ImageType::ConstPointer image)
{
  /** Typedefs. */
  using IndexType = typename ImageType::IndexType;
  using IndexValueType = typename ImageType::IndexValueType;
  using PointType = typename ImageType::PointType;

  elxout << "Loading landmarks for " << this->GetComponentLabel() << ":" << this->elxGetClassName() << "." << std::endl;

  /** Read the landmarks. */
  auto reader = itk::TransformixInputPointFileReader<PointSetType>::New();
  reader->SetFileName(landmarkFileName.c_str());
  elxout << "  Reading landmark file: " << landmarkFileName << std::endl;
  try
  {
    reader->Update();
  }
  catch (const itk::ExceptionObject & err)
  {
    xl::xout["error"] << "  Error while opening " << landmarkFileName << std::endl;
    xl::xout["error"] << err << std::endl;
    itkExceptionMacro(<< "ERROR: unable to configure " << this->GetComponentLabel());
  }

  /** Some user-feedback. */
  const unsigned int nrofpoints = reader->GetNumberOfPoints();
  if (reader->GetPointsAreIndices())
  {
    elxout << "  Landmarks are specified as image indices." << std::endl;
  }
  else
  {
    elxout << "  Landmarks are specified in world coordinates." << std::endl;
  }
  elxout << "  Number of specified points: " << nrofpoints << std::endl;

  /** Get the pointset. */
  pointSet = reader->GetOutput();

  /** Convert from index to point if necessary */
  pointSet->DisconnectPipeline();
  if (reader->GetPointsAreIndices())
  {
    /** Convert to world coordinates */
    for (unsigned int j = 0; j < nrofpoints; ++j)
    {
      /** The landmarks from the pointSet are indices. We first cast to the
       * proper type, and then convert it to world coordinates.
       */
      PointType point;
      IndexType index;
      pointSet->GetPoint(j, &point);
      for (unsigned int d = 0; d < FixedImageDimension; ++d)
      {
        index[d] = static_cast<IndexValueType>(vnl_math::rnd(point[d]));
      }

      /** Compute the input point in physical coordinates. */
      image->TransformIndexToPhysicalPoint(index, point);
      pointSet->SetPoint(j, point);

    } // end for all points
  }   // end for points are indices

  return nrofpoints;

} // end ReadLandmarks()


/**
 * ************** TransformPointsSomePointsVTK *********************
 */

template <class TElastix>
unsigned int
StatisticalShapePenalty<TElastix>::ReadShape(const std::string &                    ShapeFileName,
                                             typename PointSetType::Pointer &       pointSet,
                                             const typename ImageType::ConstPointer image)
{
  /** Typedef's. \todo test DummyIPPPixelType=bool. */
  using DummyIPPPixelType = double;
  using MeshTraitsType =
    DefaultStaticMeshTraits<DummyIPPPixelType, FixedImageDimension, FixedImageDimension, CoordRepType>;
  using MeshType = Mesh<DummyIPPPixelType, FixedImageDimension, MeshTraitsType>;

  /** Read the input points. */
  auto meshReader = MeshFileReader<MeshType>::New();
  meshReader->SetFileName(ShapeFileName.c_str());
  elxout << "  Reading input point file: " << ShapeFileName << std::endl;
  try
  {
    meshReader->Update();
  }
  catch (ExceptionObject & err)
  {
    xl::xout["error"] << "  Error while opening input point file." << std::endl;
    xl::xout["error"] << err << std::endl;
  }

  /** Some user-feedback. */
  elxout << "  Input points are specified in world coordinates." << std::endl;
  unsigned long nrofpoints = meshReader->GetOutput()->GetNumberOfPoints();
  elxout << "  Number of specified input points: " << nrofpoints << std::endl;

  typename MeshType::Pointer mesh = meshReader->GetOutput();
  pointSet = PointSetType::New();
  pointSet->SetPoints(mesh->GetPoints());
  return nrofpoints;

} // end ReadShape()


} // end namespace elastix

#endif // end #ifndef elxStatisticalShapePenalty_hxx
