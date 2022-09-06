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
#ifndef itkVectorContainerSource_h
#define itkVectorContainerSource_h

#include "itkProcessObject.h"
#include "itkDataObjectDecorator.h"

namespace itk
{
/** \class VectorContainerSource
 *
 * \brief A base class for creating an ImageToVectorContainerFilter.
 */

template <class TOutputVectorContainer>
class ITK_TEMPLATE_EXPORT VectorContainerSource : public ProcessObject
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(VectorContainerSource);

  /** Standard ITK-stuff. */
  using Self = VectorContainerSource;
  using Superclass = ProcessObject;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(VectorContainerSource, ProcessObject);

  /** Some convenient typedefs. */
  using typename Superclass::DataObjectPointer;
  using OutputVectorContainerType = TOutputVectorContainer;
  using OutputVectorContainerPointer = typename OutputVectorContainerType::Pointer;

  /** Get the vector container output of this process object. */
  OutputVectorContainerType *
  GetOutput();

  /** Get the vector container output of this process object. */
  OutputVectorContainerType *
  GetOutput(unsigned int idx);

  /** Graft the specified DataObject onto this ProcessObject's output. */
  virtual void
  GraftOutput(DataObject * output);

  /** Graft the specified DataObject onto this ProcessObject's output. */
  virtual void
  GraftNthOutput(unsigned int idx, DataObject * output);

  /** Make a DataObject of the correct type to used as the specified output. */
  virtual DataObjectPointer
  MakeOutput(unsigned int idx);

protected:
  /** The constructor. */
  VectorContainerSource();
  /** The destructor. */
  ~VectorContainerSource() override = default;

  /** PrintSelf. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

private:
  /** Member variables. */
  int m_GenerateDataRegion;
  int m_GenerateDataNumberOfRegions;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkVectorContainerSource.hxx"
#endif

#endif // end #ifndef itkVectorContainerSource_h
