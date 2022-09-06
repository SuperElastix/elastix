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
#ifndef itkMeshFileReaderBase_h
#define itkMeshFileReaderBase_h

#include "itkMeshSource.h"
#include "itkMacro.h"
#include "itkMeshFileReader.h" // for MeshFileReaderException

namespace itk
{

/** \class MeshFileReaderBase
 *
 * \brief Base class for mesh readers
 *
 * A base class for classes that read a file containing
 * a mesh or a pointset.
 */

template <class TOutputMesh>
class ITK_TEMPLATE_EXPORT MeshFileReaderBase : public MeshSource<TOutputMesh>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(MeshFileReaderBase);

  /** Standard class typedefs. */
  using Self = MeshFileReaderBase;
  using Superclass = MeshSource<TOutputMesh>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MeshFileReaderBase, MeshSource);

  /** Some convenient typedefs. */
  using typename Superclass::DataObjectPointer;
  using typename Superclass::OutputMeshType;
  using typename Superclass::OutputMeshPointer;

  /** Set/Get the filename */
  itkGetStringMacro(FileName);
  itkSetStringMacro(FileName);

  /** Prepare the allocation of the output mesh during the first back
   * propagation of the pipeline.
   */
  void
  GenerateOutputInformation() override;

  /** Give the reader a chance to indicate that it will produce more
   * output than it was requested to produce. MeshFileReader cannot
   * currently read a portion of a mesh, so the MeshFileReader must
   * enlarge the RequestedRegion to the size of the mesh on disk.
   */
  void
  EnlargeOutputRequestedRegion(DataObject * output) override;

protected:
  MeshFileReaderBase() = default;
  ~MeshFileReaderBase() override = default;

  /** Test whether the given filename exist and it is readable,
   * this is intended to be called before attempting to use
   * subclasses for actually reading the file. If the file
   * doesn't exist or it is not readable, and exception with an
   * appropriate message will be thrown.
   */
  virtual void
  TestFileExistanceAndReadability();

  std::string m_FileName{};
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkMeshFileReaderBase.hxx"
#endif

#endif
