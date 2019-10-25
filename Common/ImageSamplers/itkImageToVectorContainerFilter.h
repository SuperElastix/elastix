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
#ifndef __itkImageToVectorContainerFilter_h
#define __itkImageToVectorContainerFilter_h

#include "itkVectorContainerSource.h"
#include "itkPlatformMultiThreader.h"

namespace itk
{

/** \class ImageToVectorContainerFilter
 *
 * \brief Base class that takes in an image and pops out
 * a vector container.
 */

template< class TInputImage, class TOutputVectorContainer >
class ImageToVectorContainerFilter :
  public VectorContainerSource< TOutputVectorContainer >
{
public:

  /** Standard ITK-stuff. */
  typedef ImageToVectorContainerFilter                    Self;
  typedef VectorContainerSource< TOutputVectorContainer > Superclass;
  typedef SmartPointer< Self >                            Pointer;
  typedef SmartPointer< const Self >                      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( ImageToVectorContainerFilter, VectorContainerSource );

  /** Typedefs inherited from the superclass. */
  typedef typename Superclass::DataObjectPointer            DataObjectPointer;
  typedef typename Superclass::OutputVectorContainerType    OutputVectorContainerType;
  typedef typename Superclass::OutputVectorContainerPointer OutputVectorContainerPointer;

  /** Some Image related typedefs. */
  typedef TInputImage                           InputImageType;
  typedef typename InputImageType::Pointer      InputImagePointer;
  typedef typename InputImageType::ConstPointer InputImageConstPointer;
  typedef typename InputImageType::RegionType   InputImageRegionType;
  typedef typename InputImageType::PixelType    InputImagePixelType;

  /** Create a valid output. */
  DataObject::Pointer MakeOutput( unsigned int idx ) override;

  /** Set the input image of this process object.  */
  void SetInput( unsigned int idx, const InputImageType * input );

  /** Set the input image of this process object.  */
  void SetInput( const InputImageType * input );

  /** Get the input image of this process object.  */
  const InputImageType * GetInput( void );

  /** Get the input image of this process object.  */
  const InputImageType * GetInput( unsigned int idx );

  /** Get the output Mesh of this process object.  */
  OutputVectorContainerType * GetOutput( void );

protected:

  /** A version of GenerateData() specific for image processing
   * filters.  This implementation will split the processing across
   * multiple threads. The buffer is allocated by this method. Then
   * the BeforeThreadedGenerateData() method is called (if
   * provided). Then, a series of threads are spawned each calling
   * ThreadedGenerateData(). After all the threads have completed
   * processing, the AfterThreadedGenerateData() method is called (if
   * provided). If an image processing filter cannot be threaded, the
   * filter should provide an implementation of GenerateData(). That
   * implementation is responsible for allocating the output buffer.
   * If a filter an be threaded, it should NOT provide a
   * GenerateData() method but should provide a ThreadedGenerateData()
   * instead.
   *
   * \sa ThreadedGenerateData() */
  void GenerateData( void ) override;

  /** If an imaging filter can be implemented as a multithreaded
   * algorithm, the filter will provide an implementation of
   * DynamicThreadedGenerateData().  This superclass will automatically split
   * the output image into a number of pieces
   * and call DynamicThreadedGenerateData() in each work unit. Prior to
   * parallel
   * execution, the BeforeThreadedGenerateData() method is called. After
   * all the work units have completed, the AfterThreadedGenerateData()
   * method is called. If an image processing filter cannot support
   * threading, that filter should provide an implementation of the
   * GenerateData() method instead of providing an implementation of
   * DynamicThreadedGenerateData().  If a filter provides a GenerateData()
   * method as its implementation, then the filter is responsible for
   * allocating the output data.  If a filter provides a
   * ThreadedGenerateData() method as its implementation, then the
   * output memory will allocated automatically by this superclass.
   * The ThreadedGenerateData() method should only produce the output
   * specified by "outputThreadRegion"
   * parameter. DynamicThreadedGenerateData() cannot write to any other
   * portion of the output image (as this is responsibility of a
   * different thread).
   *
   * \sa GenerateData() */
  virtual void DynamicThreadedGenerateData( const InputImageRegionType & inputRegionForWorkUnit );

  /** If an imaging filter needs to perform processing after the buffer
   * has been allocated but before threads are spawned, the filter can
   * can provide an implementation for BeforeThreadedGenerateData(). The
   * execution flow in the default GenerateData() method will be:
   *      1) Allocate the output buffer
   *      2) Call BeforeThreadedGenerateData()
   *      3) Spawn threads, calling ThreadedGenerateData() in each thread.
   *      4) Call AfterThreadedGenerateData()
   * Note that this flow of control is only available if a filter provides
   * a ThreadedGenerateData() method and NOT a GenerateData() method. */
  virtual void BeforeThreadedGenerateData( void ) {}

  /** If an imaging filter needs to perform processing after all
   * processing threads have completed, the filter can can provide an
   * implementation for AfterThreadedGenerateData(). The execution
   * flow in the default GenerateData() method will be:
   *      1) Allocate the output buffer
   *      2) Call BeforeThreadedGenerateData()
   *      3) Spawn threads, calling ThreadedGenerateData() in each thread.
   *      4) Call AfterThreadedGenerateData()
   * Note that this flow of control is only available if a filter provides
   * a ThreadedGenerateData() method and NOT a GenerateData() method. */
  virtual void AfterThreadedGenerateData( void ) {}


  /** The constructor. */
  ImageToVectorContainerFilter();
  /** The destructor. */
  ~ImageToVectorContainerFilter() override {}

  /** PrintSelf. */
  void PrintSelf( std::ostream & os, Indent indent ) const override;

private:

  /** The private constructor. */
  ImageToVectorContainerFilter( const Self & );  // purposely not implemented
  /** The private copy constructor. */
  void operator=( const Self & );                // purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageToVectorContainerFilter.hxx"
#endif

#endif // end #ifndef __itkImageToVectorContainerFilter_h
