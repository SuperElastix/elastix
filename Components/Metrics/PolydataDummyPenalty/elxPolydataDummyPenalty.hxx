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
#ifndef elxPolydataDummyPenalty_hxx
#define elxPolydataDummyPenalty_hxx

#include <typeinfo>

namespace elastix
{

/**
 * ******************* Constructor *******************
 */

template <typename TElastix>
PolydataDummyPenalty<TElastix>::PolydataDummyPenalty()
{
  this->m_NumberOfMeshes = 0;
}


/**
 * ******************* Initialize ***********************
 */

template <typename TElastix>
void
PolydataDummyPenalty<TElastix>::Initialize()
{
  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();
  timer.Stop();
  log::info(std::ostringstream{} << "Initialization of PolydataDummyPenalty metric took: "
                                 << static_cast<long>(timer.GetMean() * 1000) << " ms.");

} // end Initialize()


/**
 * ***************** BeforeAllBase ***********************
 */

template <typename TElastix>
int
PolydataDummyPenalty<TElastix>::BeforeAllBase()
{
  this->Superclass2::BeforeAllBase();

  /** Check if the current configuration uses this metric. */
  unsigned int count = 0;
  for (unsigned int i = 0; i < this->m_Configuration->CountNumberOfParameterEntries("Metric"); ++i)
  {
    std::string metricName = "";
    this->m_Configuration->ReadParameter(metricName, "Metric", i);
    if (metricName == this->elxGetClassName())
    {
      ++count;
    }
  }
  if (count == 0)
  {
    return 0;
  }

  std::string componentLabel(this->GetComponentLabel());
  std::string metricNumber = componentLabel.substr(6, 2); // strip "Metric" keep number

  /** Check Command line options and print them to the log file. */
  log::info(std::ostringstream{} << "Command line options from " << this->elxGetClassName() << ": (" << componentLabel
                                 << "):");
  this->m_NumberOfMeshes = 0;

  for (char ch = 'A'; ch <= 'Z'; ++ch)
  {
    std::ostringstream fmeshArgument("-fmesh", std::ios_base::ate);
    /** Check for appearance of "-fmesh<[A-Z]><Metric>". */
    fmeshArgument << ch << metricNumber;

    if (const std::string commandLineArgument = this->m_Configuration->GetCommandLineArgument(fmeshArgument.str());
        commandLineArgument.empty())
    {
      break;
    }
    else
    {
      log::info(fmeshArgument.str() + "\t" + commandLineArgument);
      this->m_NumberOfMeshes++;
    }
  }
  /** Return a value. */
  return 0;

} // end BeforeAllBase()


/**
 * ***************** BeforeRegistration ***********************
 */

template <typename TElastix>
void
PolydataDummyPenalty<TElastix>::BeforeRegistration()
{
  std::string componentLabel(this->GetComponentLabel());
  std::string metricNumber = componentLabel.substr(6, 2); // strip "Metric" keep number

  log::info(std::ostringstream{} << "Loading meshes for " << this->GetComponentLabel() << ":" << this->elxGetClassName()
                                 << ".");

  FixedMeshContainerPointer meshPointerContainer = FixedMeshContainerType::New();
  meshPointerContainer->Reserve(this->m_NumberOfMeshes);
  unsigned int meshNumber;
  char         ch;
  for (meshNumber = 0, ch = 'A'; meshNumber < this->m_NumberOfMeshes; ++meshNumber, ++ch)
  {
    std::ostringstream fmeshArgument("-fmesh", std::ios_base::ate);
    fmeshArgument << ch << metricNumber;
    std::string                fixedMeshName = this->GetConfiguration()->GetCommandLineArgument(fmeshArgument.str());
    typename MeshType::Pointer fixedMesh; // default-constructed (null)
    if (itksys::SystemTools::GetFilenameLastExtension(fixedMeshName) == ".txt")
    {
      this->ReadTransformixPoints(fixedMeshName, fixedMesh);
    }
    else
    {
      this->ReadMesh(fixedMeshName, fixedMesh);
    }

    meshPointerContainer->SetElement(meshNumber, fixedMesh.GetPointer());
  }

  this->SetFixedMeshContainer(meshPointerContainer);

  auto dummyPointSet = PointSetType::New();
  this->SetFixedPointSet(dummyPointSet);  // FB: TODO solve hack
  this->SetMovingPointSet(dummyPointSet); // FB: TODO solve hack
  // itkCombinationImageToImageMetric.hxx checks if metric base class is ImageMetricType or PointSetMetricType.
  // This class is derived from SingleValuedPointSetToPointSetMetric which needs a moving pointset.
  // Without interfering with some general elastix files, this hack gives me the functionality that I needed.
  // TODO: let itkCombinationImageToImageMetric check for a base class metric that doesn't use an image or moving
  // pointset.

} // end BeforeRegistration()


/**
 * ***************** AfterEachIteration ***********************
 */

template <typename TElastix>
void
PolydataDummyPenalty<TElastix>::AfterEachIteration()
{
  /** What is the current resolution level? */
  const unsigned int level = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

  /** What is the current iteration number? */
  const unsigned int iter = this->m_Elastix->GetIterationCounter();

  /** Decide whether or not to write the result mesh this iteration. */
  bool writeResultMeshThisIteration = false;
  this->m_Configuration->ReadParameter(
    writeResultMeshThisIteration, "WriteResultMeshAfterEachIteration", "", level, 0, false);

  const std::string outputDirectoryPath = this->m_Configuration->GetCommandLineArgument("-out");

  /** Writing result mesh. */
  if (writeResultMeshThisIteration && !outputDirectoryPath.empty())
  {
    std::string componentLabel(this->GetComponentLabel());
    std::string metricNumber = componentLabel.substr(6, 2); // strip "Metric" keep number

    /** Create a name for the final result. */
    std::string resultMeshFormat = "vtk";
    this->m_Configuration->ReadParameter(resultMeshFormat, "ResultMeshFormat", 0, false);
    char ch = 'A';
    for (MeshIdType meshId = 0; meshId < this->m_NumberOfMeshes; ++meshId, ++ch)
    {

      std::ostringstream makeFileName;
      makeFileName << outputDirectoryPath << "resultmesh" << ch << metricNumber << "."
                   << this->m_Configuration->GetElastixLevel() << ".R" << level << ".It" << std::setfill('0')
                   << std::setw(7) << iter << "." << resultMeshFormat;

      try
      {
        this->WriteResultMesh(makeFileName.str(), meshId);
      }
      catch (const itk::ExceptionObject & excp)
      {
        log::error(std::ostringstream{} << "Exception caught: \n" << excp << "Resuming elastix.");
      }
    } // end for
  }   // end if

} // end AfterEachIteration()


/**
 * ***************** AfterEachResolution ***********************
 */

template <typename TElastix>
void
PolydataDummyPenalty<TElastix>::AfterEachResolution()
{
  /** What is the current resolution level? */
  const unsigned int level = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

  /** Decide whether or not to write the result mesh this resolution. */
  bool writeResultMeshThisResolution = false;
  this->m_Configuration->ReadParameter(
    writeResultMeshThisResolution, "WriteResultMeshAfterEachResolution", "", level, 0, false);

  const std::string outputDirectoryPath = this->m_Configuration->GetCommandLineArgument("-out");

  /** Writing result mesh. */
  if (writeResultMeshThisResolution && !outputDirectoryPath.empty())
  {
    std::string componentLabel(this->GetComponentLabel());
    std::string metricNumber = componentLabel.substr(6, 2); // strip "Metric" keep number

    /** Create a name for the final result. */
    std::string resultMeshFormat = "vtk";
    this->m_Configuration->ReadParameter(resultMeshFormat, "ResultMeshFormat", 0, false);
    char ch = 'A';
    for (MeshIdType meshId = 0; meshId < this->m_NumberOfMeshes; ++meshId, ++ch)
    {

      std::ostringstream makeFileName;
      makeFileName << outputDirectoryPath << "resultmesh" << ch << metricNumber << "."
                   << this->m_Configuration->GetElastixLevel() << ".R" << level << "." << resultMeshFormat;

      try
      {
        this->WriteResultMesh(makeFileName.str(), meshId);
      }
      catch (const itk::ExceptionObject & excp)
      {
        log::error(std::ostringstream{} << "Exception caught: \n" << excp << "Resuming elastix.");
      }
    } // end for
  }   // end if

} // end AfterEachResolution()


/**
 * ************** ReadMesh *********************
 */

template <typename TElastix>
unsigned int
PolydataDummyPenalty<TElastix>::ReadMesh(const std::string & meshFileName, typename FixedMeshType::Pointer & mesh)
{
  /** Read the input mesh. */
  auto meshReader = itk::MeshFileReader<MeshType>::New();
  meshReader->SetFileName(meshFileName);

  log::info(std::ostringstream{} << "  Reading input mesh file: " << meshFileName);
  try
  {
    // meshReader->Update();
    meshReader->UpdateLargestPossibleRegion();
  }
  catch (const itk::ExceptionObject & err)
  {
    log::error(std::ostringstream{} << "  Error while opening input mesh file.\n" << err);
  }

  /** Some user-feedback. */
  mesh = meshReader->GetOutput();
  unsigned long nrofpoints = mesh->GetNumberOfPoints();
  log::info(std::ostringstream{} << "  Number of specified input points: " << nrofpoints);

  return nrofpoints;
} // end ReadMesh()


/**
 * ******************* WriteResultMesh ********************
 */

template <typename TElastix>
void
PolydataDummyPenalty<TElastix>::WriteResultMesh(const std::string & filename, MeshIdType meshId)
{
  /** Set the points of the latest transformation. */
  const MappedMeshContainerPointer mappedMeshContainer = this->GetModifiableMappedMeshContainer();
  FixedMeshPointer                 mappedMesh = mappedMeshContainer->ElementAt(meshId);

  /** Use pointer to the mesh data of fixedMesh; const_cast are assumed since outputMesh
   * will only be used for writing the output.
   * */
  FixedMeshConstPointer fixedMesh = this->GetFixedMeshContainer()->ElementAt(meshId);
  bool                  tempSetPointData = (mappedMesh->GetPointData() == nullptr);
  bool                  tempSetCells = (mappedMesh->GetCells() == nullptr);
  bool                  tempSetCellData = (mappedMesh->GetCellData() == nullptr);

  if (tempSetPointData)
  {
    mappedMesh->SetPointData(const_cast<typename MeshType::PointDataContainer *>(fixedMesh->GetPointData()));
  }

  if (tempSetCells)
  {
    mappedMesh->SetCells(const_cast<typename MeshType::CellsContainer *>(fixedMesh->GetCells()));
  }
  if (tempSetCellData)
  {
    mappedMesh->SetCellData(const_cast<typename MeshType::CellDataContainer *>(fixedMesh->GetCellData()));
  }

  mappedMesh->Modified();
  mappedMesh->Update();

  try
  {
    itk::WriteMesh(mappedMesh, filename);
  }
  catch (itk::ExceptionObject & excp)
  {
    /** Add information to the exception. */
    excp.SetLocation("PolydataDummyPenalty - WriteResultMesh()");
    std::string err_str = excp.GetDescription();
    err_str += "\nError occurred while writing mapped mesh.\n";
    excp.SetDescription(err_str);

    /** Pass the exception to an higher level. */
    throw;
  }

  if (tempSetPointData)
  {
    mappedMesh->SetPointData(nullptr);
  }

  if (tempSetCells)
  {
    mappedMesh->SetCells(nullptr);
  }
  if (tempSetCellData)
  {
    mappedMesh->SetCellData(nullptr);
  }

} // end WriteResultMesh()


/**
 * ******************* ReadTransformixPoints ********************
 */

template <typename TElastix>
unsigned int
PolydataDummyPenalty<TElastix>::ReadTransformixPoints(const std::string &          filename,
                                                      typename MeshType::Pointer & mesh) // const
{
  /*
  Floris: Mainly copied from elxTransformBase.hxx
  */
  /** Typedef's. */
  using FixedImageRegionType = typename FixedImageType::RegionType;
  using FixedImageOriginType = typename FixedImageType::PointType;
  using FixedImageSpacingType = typename FixedImageType::SpacingType;
  using FixedImageIndexType = typename FixedImageType::IndexType;
  using FixedImageIndexValueType = typename FixedImageIndexType::IndexValueType;
  using MovingImageIndexType = typename MovingImageType::IndexType;
  using FixedImageDirectionType = typename FixedImageType::DirectionType;

  using DummyIPPPixelType = unsigned char;
  using DummyMeshTraitsType =
    itk::DefaultStaticMeshTraits<DummyIPPPixelType, FixedImageDimension, FixedImageDimension, CoordinateType>;
  using DummyPointSetType = itk::PointSet<DummyIPPPixelType, FixedImageDimension, DummyMeshTraitsType>;
  using DeformationVectorType = itk::Vector<float, FixedImageDimension>;

  /** Construct an ipp-file reader. */
  auto ippReader = itk::TransformixInputPointFileReader<DummyPointSetType>::New();
  ippReader->SetFileName(filename);

  /** Read the input points. */
  log::info(std::ostringstream{} << "  Reading input point file: " << filename);
  try
  {
    ippReader->Update();
  }
  catch (const itk::ExceptionObject & err)
  {
    log::error(std::ostringstream{} << "  Error while opening input point file.\n" << err);
  }

  /** Some user-feedback. */
  if (ippReader->GetPointsAreIndices())
  {
    log::info("  Input points are specified as image indices.");
  }
  else
  {
    log::info("  Input points are specified in world coordinates.");
  }
  const unsigned int nrofpoints = ippReader->GetNumberOfPoints();
  log::info(std::ostringstream{} << "  Number of specified input points: " << nrofpoints);

  /** Get the set of input points. */
  typename DummyPointSetType::Pointer inputPointSet = ippReader->GetOutput();

  /** Create the storage classes. */
  std::vector<FixedImageIndexType>   inputindexvec(nrofpoints);
  std::vector<InputPointType>        inputpointvec(nrofpoints);
  std::vector<FixedImageIndexType>   outputindexfixedvec(nrofpoints);
  std::vector<MovingImageIndexType>  outputindexmovingvec(nrofpoints);
  std::vector<DeformationVectorType> deformationvec(nrofpoints);

  const auto & resampleImageFilter = *(this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType());

  /** Make a temporary image with the right region info,
   * which we can use to convert between points and indices.
   * By taking the image from the resampler output, the UseDirectionCosines
   * parameter is automatically taken into account.
   */
  auto dummyImage = FixedImageType::New();
  dummyImage->SetRegions(
    FixedImageRegionType(resampleImageFilter.GetOutputStartIndex(), resampleImageFilter.GetSize()));
  dummyImage->SetOrigin(resampleImageFilter.GetOutputOrigin());
  dummyImage->SetSpacing(resampleImageFilter.GetOutputSpacing());
  dummyImage->SetDirection(resampleImageFilter.GetOutputDirection());

  /** Read the input points, as index or as point. */
  if (ippReader->GetPointsAreIndices())
  {
    for (unsigned int j = 0; j < nrofpoints; ++j)
    {
      /** The read point from the inutPointSet is actually an index
       * Cast to the proper type.
       */
      InputPointType point{};
      inputPointSet->GetPoint(j, &point);
      for (unsigned int i = 0; i < FixedImageDimension; ++i)
      {
        inputindexvec[j][i] = static_cast<FixedImageIndexValueType>(vnl_math::rnd(point[i]));
      }
      /** Compute the input point in physical coordinates. */
      dummyImage->TransformIndexToPhysicalPoint(inputindexvec[j], inputpointvec[j]);
    }
  }
  else
  {
    for (unsigned int j = 0; j < nrofpoints; ++j)
    {
      /** Compute index of nearest voxel in fixed image. */
      InputPointType point{};
      inputPointSet->GetPoint(j, &point);
      inputpointvec[j] = point;
      const auto fixedcindex = dummyImage->template TransformPhysicalPointToContinuousIndex<double>(point);
      for (unsigned int i = 0; i < FixedImageDimension; ++i)
      {
        inputindexvec[j][i] = static_cast<FixedImageIndexValueType>(vnl_math::rnd(fixedcindex[i]));
      }
    }
  }
  /** Floris: create a mesh containing the points**/
  mesh = MeshType::New();
  mesh->SetPoints(inputPointSet->GetPoints());

  /** Floris: make connected mesh (polygon) if data is 2d by assuming the sequence of points being connected**/
  if constexpr (FixedImageDimension == 2)
  {
    using CellAutoPointer = typename MeshType::CellType::CellAutoPointer;
    using LineType = itk::LineCell<typename MeshType::CellType>;

    for (unsigned int i = 0; i < nrofpoints; ++i)
    {

      // Create a link to the previous point in the column (below the current point)
      CellAutoPointer line;
      line.TakeOwnership(new LineType);

      line->SetPointId(0, i); // line between points 0 and 1
      line->SetPointId(1, (i + 1) % nrofpoints);
      mesh->SetCell(i, line);
    }
  }

  std::cout << "mesh->GetNumberOfCells()" << mesh->GetNumberOfCells() << '\n'
            << "mesh->GetNumberOfPoints()" << mesh->GetNumberOfPoints() << std::endl;
  typename MeshType::PointsContainer::ConstPointer points = mesh->GetPoints();

  typename MeshType::PointsContainerConstIterator pointsBegin = points->Begin();
  typename MeshType::PointsContainerConstIterator pointsEnd = points->End();
  for (; pointsBegin != pointsEnd; ++pointsBegin)
  {
    std::cout << "point " << pointsBegin->Index() << ": " << pointsBegin->Value().GetVnlVector() << std::endl;
  }

  using CellIterator = typename MeshType::CellsContainer::Iterator;
  CellIterator cellIterator = mesh->GetCells()->Begin();
  CellIterator CellsEnd = mesh->GetCells()->End();

  typename CellInterfaceType::PointIdIterator beginpointer;
  typename CellInterfaceType::PointIdIterator endpointer;

  for (; cellIterator != CellsEnd; ++cellIterator)
  {
    std::cout << "Cell Index: " << cellIterator->Index() << std::endl;
    beginpointer = cellIterator->Value()->PointIdsBegin();
    endpointer = cellIterator->Value()->PointIdsEnd();

    for (; beginpointer != endpointer; ++beginpointer)
    {
      std::cout << "Id: " << *beginpointer << std::endl;
    }
  }
  return nrofpoints;
} // end ReadTransformixPoints()


} // end namespace elastix

#endif // end #ifndef elxPolydataDummyPenalty_hxx
