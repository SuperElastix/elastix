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
#ifndef itkMeshFileReaderBase_hxx
#define itkMeshFileReaderBase_hxx

#include "itkMeshFileReaderBase.h"

#include <itksys/SystemTools.hxx>
#include <fstream>

namespace itk
{

/**
 * ***************GenerateOutputInformation ***********
 */

template <class TOutputMesh>
void
MeshFileReaderBase<TOutputMesh>::GenerateOutputInformation()
{
  OutputMeshPointer output = this->GetOutput();

  itkDebugMacro(<< "Reading file for GenerateOutputInformation(): " << m_FileName);

  /** Check to see if we can read the file given the name or prefix */
  if (m_FileName.empty())
  {
    throw MeshFileReaderException(__FILE__, __LINE__, "FileName must be specified", ITK_LOCATION);
  }

  /** Test if the file exist and if it can be open.
   * and exception will be thrown otherwise. */
  this->TestFileExistanceAndReadability();

  // Copy MetaDataDictionary from instantiated reader to output mesh?
  // output->SetMetaDataDictionary(m_ImageIO->GetMetaDataDictionary());
  // this->SetMetaDataDictionary(m_ImageIO->GetMetaDataDictionary());

  // This makes not really sense i think.
  // MeshRegionType region;
  // region = ?
  // output->SetLargestPossibleRegion(region);

} // end GenerateOutputInformation()


/**
 * *************TestFileExistanceAndReadability ***********
 */

template <class TOutputMesh>
void
MeshFileReaderBase<TOutputMesh>::TestFileExistanceAndReadability()
{
  // Test if the file exists.
  if (!itksys::SystemTools::FileExists(m_FileName))
  {
    std::ostringstream msg;
    msg << "The file doesn't exists. \nFilename = " << m_FileName << '\n';
    throw MeshFileReaderException(__FILE__, __LINE__, msg.str().c_str(), ITK_LOCATION);
  }

  // Test if the file can be open for reading access.
  const std::ifstream readTester(m_FileName);
  if (readTester.fail())
  {
    std::ostringstream msg;
    msg << "The file couldn't be opened for reading. \nFilename: " << m_FileName << '\n';
    throw MeshFileReaderException(__FILE__, __LINE__, msg.str().c_str(), ITK_LOCATION);
  }

} // end TestFileExistanceAndReadability()


/**
 * **************EnlargeOutputRequestedRegion***********
 */

template <class TOutputMesh>
void
MeshFileReaderBase<TOutputMesh>::EnlargeOutputRequestedRegion(DataObject * output)
{
  OutputMeshPointer out = dynamic_cast<OutputMeshType *>(output);

  if (out)
  {
    out->SetRequestedRegionToLargestPossibleRegion();
  }
  else
  {
    throw MeshFileReaderException(__FILE__, __LINE__, "Invalid output object type");
  }

} // end EnlargeOutputRequestedRegion()


} // end namespace itk

#endif
