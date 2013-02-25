/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUAdvancedMatrixOffsetTransformBase_h
#define __itkGPUAdvancedMatrixOffsetTransformBase_h

#include "itkAdvancedMatrixOffsetTransformBase.h"
#include "itkGPUMatrixOffsetTransformBase.h"

namespace itk
{
/** \class GPUAdvancedMatrixOffsetTransformBase
* \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
* Department of Radiology, Leiden, The Netherlands
*
* This implementation was taken from elastix (http://elastix.isi.uu.nl/).
*
* \note This work was funded by the Netherlands Organisation for
* Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
*
*/
template< class TScalarType = float, unsigned int NDimensions = 3,
          class TParentImageFilter = AffineTransform< TScalarType, NDimensions > >
class GPUAdvancedMatrixOffsetTransformBase :
  public TParentImageFilter,
  public GPUMatrixOffsetTransformBase< TScalarType, NDimensions, NDimensions >
{
public:
  /** Standard class typedefs. */
  typedef GPUAdvancedMatrixOffsetTransformBase Self;
  typedef TParentImageFilter                   CPUSuperclass;
  typedef GPUMatrixOffsetTransformBase< TScalarType,
                                        NDimensions,
                                        NDimensions > GPUSuperclass;

  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUAdvancedMatrixOffsetTransformBase, CPUSuperclass );

  /** Typedefs */
  typedef typename GPUSuperclass::CPUMatrixType        CPUMatrixType;
  typedef typename GPUSuperclass::CPUInverseMatrixType CPUInverseMatrixType;
  typedef typename GPUSuperclass::CPUOutputVectorType  CPUOutputVectorType;

  /**  */
  virtual const CPUMatrixType &        GetCPUMatrix( void ) const { return this->GetMatrix(); }
  virtual const CPUInverseMatrixType & GetCPUInverseMatrix( void ) const { return this->GetInverseMatrix(); }
  virtual const CPUOutputVectorType &  GetCPUOffset( void ) const { return this->GetOffset(); }

protected:
  GPUAdvancedMatrixOffsetTransformBase() {}
  virtual ~GPUAdvancedMatrixOffsetTransformBase() {}

private:
  GPUAdvancedMatrixOffsetTransformBase( const Self & other ); // purposely not implemented
  const Self & operator=( const Self & );                     // purposely not implemented
};

/** \class GPUAdvancedMatrixOffsetTransformBaseFactory
* \brief Object Factory implementation for GPUAdvancedMatrixOffsetTransformBase
*/
class GPUAdvancedMatrixOffsetTransformBaseFactory : public ObjectFactoryBase
{
public:
  typedef GPUAdvancedMatrixOffsetTransformBaseFactory Self;
  typedef ObjectFactoryBase                           Superclass;
  typedef SmartPointer< Self >                        Pointer;
  typedef SmartPointer< const Self >                  ConstPointer;

  /** Class methods used to interface with the registered factories. */
  virtual const char * GetITKSourceVersion() const { return ITK_SOURCE_VERSION; }
  const char * GetDescription() const { return "A Factory for GPUAdvancedMatrixOffsetTransformBase"; }

  /** Method for class instantiation. */
  itkFactorylessNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUAdvancedMatrixOffsetTransformBaseFactory, itk::ObjectFactoryBase );

  /** Register one factory of this type  */
  static void RegisterOneFactory( void )
  {
    GPUAdvancedMatrixOffsetTransformBaseFactory::Pointer factory =
      GPUAdvancedMatrixOffsetTransformBaseFactory::New();
    ObjectFactoryBase::RegisterFactory( factory );
  }

private:
  GPUAdvancedMatrixOffsetTransformBaseFactory( const Self & ); // purposely not implemented
  void operator=( const Self & );                              // purposely not implemented

#define OverrideAdvancedMatrixOffsetTransformBaseTypeMacro( st, dm )                   \
  {                                                                                    \
    this->RegisterOverride(                                                            \
      typeid( AdvancedMatrixOffsetTransformBase< st, dm > ).name(),                    \
      typeid( GPUAdvancedMatrixOffsetTransformBase< st, dm > ).name(),                 \
      "GPU AdvancedMatrixOffsetTransformBase Override",                                \
      true,                                                                            \
      CreateObjectFunction< GPUAdvancedMatrixOffsetTransformBase< st, dm > >::New() ); \
  }

  GPUAdvancedMatrixOffsetTransformBaseFactory()
  {
    if ( IsGPUAvailable() )
    {
      OverrideAdvancedMatrixOffsetTransformBaseTypeMacro( float, 1 );
      //OverrideAdvancedMatrixOffsetTransformBaseTypeMacro(double, 1);

      OverrideAdvancedMatrixOffsetTransformBaseTypeMacro( float, 2 );
      //OverrideAdvancedMatrixOffsetTransformBaseTypeMacro(double, 2);

      OverrideAdvancedMatrixOffsetTransformBaseTypeMacro( float, 3 );
      //OverrideAdvancedMatrixOffsetTransformBaseTypeMacro(double, 3);
    }
  }
};
} // end namespace itk

#endif /* __itkGPUAdvancedMatrixOffsetTransformBase_h */
