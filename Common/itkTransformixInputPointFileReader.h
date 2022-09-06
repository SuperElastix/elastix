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

#ifndef itkTransformixInputPointFileReader_h
#define itkTransformixInputPointFileReader_h

#include "itkMeshFileReaderBase.h"

#include <fstream>

namespace itk
{

/** \class TransformixInputPointFileReader
 *
 * \brief A reader that understands transformix input point files
 *
 * A reader that understands transformix input point files.
 *
 * The first word in the text file indicates whether the input points
 * are given as an integer image index or as a point (world coordinates).
 * If the file does not start with "index" or "point" it is assumed
 * that the points are entered as indices, for compatibility with
 * elastix versions < 3.503.
 *
 * The second word in the text file represents the number of points that
 * should be read.
 **/

template <class TOutputMesh>
class ITK_TEMPLATE_EXPORT TransformixInputPointFileReader : public MeshFileReaderBase<TOutputMesh>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(TransformixInputPointFileReader);

  /** Standard class typedefs. */
  using Self = TransformixInputPointFileReader;
  using Superclass = MeshFileReaderBase<TOutputMesh>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(TransformixInputPointFileReader, MeshFileReaderBase);

  /** Some convenient typedefs. */
  using typename Superclass::DataObjectPointer;
  using typename Superclass::OutputMeshType;
  using typename Superclass::OutputMeshPointer;

  /** Get whether the read points are indices; actually we should store this as a kind
   * of meta data in the output, but i don't understand this concept yet...
   */
  itkGetConstMacro(PointsAreIndices, bool);

  /** Get the number of points that are defined in the file.
   * In fact we also should store this somehow in the output dataobject,
   * but that would mean resizing the point container, while still filled with
   * invalid data (since the GetNumberOfPoints method in a PointSet returns the
   * size of the point container. Storing as metadata would be another option.
   * For now leave it like this. This is a little similar to the ImageIO classes.
   * They also contain information about the image that they will read.
   * For the Mesh readers I didn't choose for a MeshIO-design, but for a
   * MeshReaderBase class and inheriting classes, so somehow it
   * seems logic to store this kind of data in the inheriting reader classes.
   */
  itkGetConstMacro(NumberOfPoints, unsigned long);

  /** Prepare the allocation of the output mesh during the first back
   * propagation of the pipeline. Updates the PointsAreIndices and NumberOfPoints.
   */
  void
  GenerateOutputInformation() override;

protected:
  TransformixInputPointFileReader() = default;
  ~TransformixInputPointFileReader() override = default;

  /** Fill the point container of the output. */
  void
  GenerateData() override;

private:
  unsigned long m_NumberOfPoints{ 0 };
  bool          m_PointsAreIndices{ false };

  std::ifstream m_Reader;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkTransformixInputPointFileReader.hxx"
#endif

#endif
