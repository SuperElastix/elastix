/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxBaseComponent_h
#define __elxBaseComponent_h

/**
 * ******************* elxBaseComponent.h *************************
 *
 * This file defines the class elx::BaseComponent, from which all
 * elastix components should inherit. It contains some methods that
 * each component is supposed to have.
 *
 * The namespace alias elx is defined in this file.
 *
 * Some header files are included that most components need.
 */


/** Get rid of warnings about too long variable names. */
#ifdef _MSC_VER
#pragma warning ( disable : 4786 )
#pragma warning ( disable : 4503 )
#endif

#include <iostream>
#include <sstream>

/** The current elastix version. */
#define __ELASTIX_VERSION 4.602

/** All elastix components should be in namespace elastix. */
namespace elastix
{

  /**
   * \class BaseComponent
   *
   * \brief The BaseComponent class is a class that all elastix
   * components should inherit from.
   *
   * Most elastix component inherit both from an ITK class and
   * from the elx::BaseComponent class. The BaseComponent class
   * contains some methods that each component is supposed
   * to have, but are not defined in itk::Object.
   *
   * \sa BaseComponentSE
   * \ingroup Install
   */

  class BaseComponent
  {
  public:

    /**
     * Callback methods that each component of elastix is supposed
     * to have. These methods can be overridden in each base component
     * (like MetricBase, TransformBase, etc.). In this way similar
     * behaviour for a group of components can be implemented.
     */
    virtual int BeforeAllBase(void) { return 0;};
    virtual int BeforeAll(void) { return 0;};

    /**
     * Callback methods that each component of elastix is supposed
     * to have. These methods can be overridden in each base component
     * (like MetricBase, TransformBase, etc.). In this way similar
     * behaviour for a group of components can be implemented.
     */
    virtual void BeforeRegistrationBase(void) {};
    virtual void BeforeEachResolutionBase(void) {};
    virtual void AfterEachResolutionBase(void) {};
    virtual void AfterEachIterationBase(void) {};
    virtual void AfterRegistrationBase(void) {};

    /**
     * Callback methods that each component of elastix is supposed
     * to have. These methods can be overridden in each single
     * component (like MattesMutualInformationMetric) to achieve
     * behaviour, specific for that single component.
     */
    virtual void BeforeRegistration(void) {};
    virtual void BeforeEachResolution(void) {};
    virtual void AfterEachResolution(void) {};
    virtual void AfterEachIteration(void) {};
    virtual void AfterRegistration(void) {};

    /**
     * The name of the component in the ComponentDatabase.
     * Override this function not directly, but with the
     * elxClassNameMacro("name").
     */
    virtual const char * elxGetClassName( void ) const
    {
      return "BaseComponent";
    }

    /** Set the component label, which consists of a label
     * ( "Metric", "Transform") and an index number. In case
     * more metrics are used simultaneously each metric will have
     * its own index number. This can be used when reading the
     * parameter file for example, to distinguish between options
     * meant for Metric0 and for Metric1.
     */
    virtual void SetComponentLabel( const char * label, unsigned int idx)
    {
      std::ostringstream makestring;
      makestring << label << idx;
      this->m_ComponentLabel = makestring.str();
    }

    /** Get the componentlabel as a string: "Metric0" for example. */
    virtual const char * GetComponentLabel( void ) const
    {
      return this->m_ComponentLabel.c_str();
    }

  protected:

    BaseComponent() {}
    virtual ~BaseComponent() {}

  private:

    BaseComponent( const BaseComponent & );   // purposely not implemented
    void operator=( const BaseComponent & );  // purposely not implemented

    std::string m_ComponentLabel;

  }; // end class BaseComponent


} // end namespace elastix

/** Define an alias for the elastix namespace.*/
namespace elx = elastix;


#endif // end #ifndef __elxBaseComponent_h
